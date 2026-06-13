"""
Unified Random Benchmarking Suite.
Includes:
1. Algorithm Comparison: L#Square vs AALpy L#
2. Unknown Sensitivity: L#Square performance across varying % of unknown outputs
"""

import sys
import random
import csv
import time
import multiprocessing
import tempfile
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aalpy.utils import (generate_random_mealy_machine, load_automaton_from_file, 
                         save_automaton_to_file, bisimilar)
from aalpy.oracles import PerfectKnowledgeEqOracle
from aalpy.learning_algs import run_Lsharp
from LSharpSquare.IncompleteMealySUL import IncompleteMealySUL, MealySUL
from LSharpSquare.IncompleteKnowledgeEqOracle import IncompleteKnowledgeEqOracle, bisimilar_with_unknowns
from LSharpSquare.LSharpSquare import run_lsharp_square

# --- Configuration Defaults ---
STATES_LIST = [10, 20, 30, 40, 50, 60]
ALPHABET_SIZES = [2, 4, 8, 16]
MACHINES_PER_CONFIG = 5
UNKNOWN_PERCENTAGES = [0, 20, 40, 60]
STATIC_SEED = 42
TIMEOUT = 120

def add_unknowns_to_machine(mealy_machine, unknown_percentage):
    """Create a copy of a Mealy Machine with a percentage of outputs replaced by 'unknown'."""
    if unknown_percentage == 0:
        return mealy_machine
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
        temp_path = f.name
    
    try:
        save_automaton_to_file(mealy_machine, temp_path)
        with open(temp_path, 'r') as f:
            dot_content = f.read()
        
        transitions = re.findall(r'label="(\d+)/(\d+|unknown)"', dot_content)
        if not transitions:
            return load_automaton_from_file(temp_path, "mealy")
        
        num_to_convert = max(1, int(len(transitions) * unknown_percentage / 100))
        indices_to_convert = set(random.sample(range(len(transitions)), num_to_convert))
        
        transition_count = 0
        def replace_output(match):
            nonlocal transition_count
            input_sym = match.group(1)
            if transition_count in indices_to_convert:
                result = f'label="{input_sym}/unknown"'
            else:
                result = match.group(0)
            transition_count += 1
            return result
        
        modified_dot = re.sub(r'label="(\d+)/(\d+|unknown)"', replace_output, dot_content)
        with open(temp_path, 'w') as f:
            f.write(modified_dot)
        
        return load_automaton_from_file(temp_path, "mealy")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_or_generate_machines(target_dir, include_unknowns=False):
    """Ensures models exist in the target directory, generating them if necessary."""
    target_dir.mkdir(exist_ok=True)
    machines = {}
    
    percentages = UNKNOWN_PERCENTAGES if include_unknowns else [0]

    if include_unknowns:
        num_states_multiplier = 10000
        alphabet_size_multiplier = 100
    else:
        num_states_multiplier = 100
        alphabet_size_multiplier = 10
    
    for num_states in STATES_LIST:
        for alphabet_size in ALPHABET_SIZES:
            for machine_id in range(1, MACHINES_PER_CONFIG + 1):
                base_machine_seed = STATIC_SEED + (num_states * num_states_multiplier + alphabet_size * alphabet_size_multiplier + machine_id)
                random.seed(base_machine_seed)
                
                base_mealy = generate_random_mealy_machine(
                    num_states=num_states,
                    input_alphabet=list(range(alphabet_size)),
                    output_alphabet=list(range(alphabet_size))
                )
                
                for p in percentages:
                    filename = f"s{num_states}_a{alphabet_size}_u{p}_m{machine_id}.dot"
                    filepath = target_dir / filename
                    
                    if filepath.exists():
                        mealy = load_automaton_from_file(str(filepath), "mealy")
                    else:
                        random.seed(base_machine_seed + p)
                        mealy = add_unknowns_to_machine(base_mealy, p)
                        save_automaton_to_file(mealy, filepath)
                    
                    machines[(num_states, alphabet_size, p, machine_id)] = mealy
    return machines

def _benchmark_worker(mealy_machine, algo_type, timeout, result_queue):
    """Worker function to run the algorithm in a separate process."""
    try:
        alphabet = list(mealy_machine.initial_state.transitions.keys())
        start_time = time.time()

        if algo_type == 'lsharp_square':
            sul = IncompleteMealySUL([], mealy_machine)
            eq_oracle = IncompleteKnowledgeEqOracle(alphabet, sul, mealy_machine)
            learned_mealy, info = run_lsharp_square(alphabet, sul, eq_oracle, return_data=True, solver_timeout=timeout)
            bisim_fn = bisimilar_with_unknowns
        else: # aalpy_lsharp
            class AalpySUL(MealySUL):
                def query(self, word: tuple):
                    self.pre()
                    for letter in word:
                        self.step(letter)
                    out = self.post()
                    self.num_queries += 1
                    self.num_successful_queries += 1
                    self.num_steps += len(word)
                    if isinstance(out, (list, tuple)) and len(out) > 0 and all(val in (None, "unknown") for val in out):
                        self.num_successful_queries -= 1
                    return out

            sul = AalpySUL(mealy_machine)
            eq_oracle = PerfectKnowledgeEqOracle(alphabet, sul, mealy_machine)
            learned_mealy, info = run_Lsharp(alphabet, sul, eq_oracle, automaton_type="mealy", return_data=True)
            bisim_fn = bisimilar
            
        elapsed = time.time() - start_time
        is_bisimilar = bisim_fn(learned_mealy, mealy_machine)
        
        result = {
            'success': is_bisimilar,
            'output_queries': info.get('queries_learning', 0),
            'equivalence_queries': info.get('validity_query' if algo_type == 'lsharp_square' else 'queries_eq_oracle', 0),
            'sul_steps': info.get('sul_steps' if algo_type == 'lsharp_square' else 'steps_learning', 0),
            'total_time': info.get('total_time', elapsed),
            'learned_states': len(learned_mealy.states)
        }
        result_queue.put(result)
    except Exception as e:
        result_queue.put({'success': False, 'error': str(e)[:100]})

def run_safe_benchmark(mealy_machine, algo_type, timeout=TIMEOUT):
    """Benchmark a machine with a hard process-level timeout."""
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_benchmark_worker, 
                                     args=(mealy_machine, algo_type, timeout, result_queue))
    process.start()
    
    # Wait for the process to finish
    process.join(timeout=timeout)
    
    if process.is_alive():
        # Handle timeout
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            'success': False, 
            'timeout': True, 
            'error': 'Timeout exceeded',
            'output_queries': '',
            'equivalence_queries': '',
            'sul_steps': '',
            'total_time': timeout,
            'learned_states': ''
        }
    
    try:
        # Retrieve result from queue
        return result_queue.get_nowait()
    except:
        return {
            'success': False, 
            'error': 'No result from process',
            'output_queries': '',
            'equivalence_queries': '',
            'sul_steps': '',
            'total_time': timeout,
            'learned_states': ''
        }

def run_comparison_benchmark(models_dir=None, results_dir=None, timeout=TIMEOUT):
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent / "models" / "randomScaling"
    else:
        models_dir = Path(models_dir)

    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results" / "randomScaling"
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    machines = get_or_generate_machines(models_dir, include_unknowns=False)
    all_results = []

    print("\n--- Running Sequential Algorithm Comparison ---")
    for (s, a, p, m_id), mealy in machines.items():
        print(f"Config S={s} A={a} M={m_id}:", end=" ", flush=True)
        ls_res = run_safe_benchmark(mealy, 'lsharp_square', timeout=timeout)
        aa_res = run_safe_benchmark(mealy, 'aalpy_lsharp', timeout=timeout)
        
        if ls_res['success'] and aa_res['success']:
            print("Completed")
            all_results.append({
                'States': s, 'Alphabet': a, 'Machine_ID': m_id,
                'L#Square_OQ': ls_res['output_queries'], 'L#Square_EQ': ls_res['equivalence_queries'],
                'L#Square_Steps': ls_res['sul_steps'], 'L#Square_Time': ls_res['total_time'],
                'L#Square_Bisimilar': ls_res['success'],
                'AALpy_OQ': aa_res['output_queries'], 'AALpy_EQ': aa_res['equivalence_queries'],
                'AALpy_Steps': aa_res['sul_steps'], 'AALpy_Time': aa_res['total_time'],
                'AALpy_Bisimilar': aa_res['success']
            })
        else:
            print("X (Failed)")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(results_dir / "comparison_raw.csv", index=False)

        # Save averaged results
        metrics = [
            'L#Square_OQ', 'L#Square_EQ', 'L#Square_Steps', 'L#Square_Time',
            'AALpy_OQ', 'AALpy_EQ', 'AALpy_Steps', 'AALpy_Time'
        ]
        res = df.groupby(['States', 'Alphabet'])[metrics].agg(['mean', 'std']).reset_index()
        res.columns = [
            f"{col[0]}_{col[1].capitalize()}" if col[1] else col[0]
            for col in res.columns.values
        ]
        res.to_csv(results_dir / "comparison_averaged.csv", index=False)

def run_unknown_benchmark(models_dir=None, results_dir=None, timeout=TIMEOUT):
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent / "models" / "randomScalingUnknown"
    else:
        models_dir = Path(models_dir)

    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results" / "randomScalingUnknown"
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    machines = get_or_generate_machines(models_dir, include_unknowns=True)
    raw_data = []

    print("\n--- Running Sequential Unknown Sensitivity Benchmark ---")
    configs = defaultdict(list)
    for key, mealy in machines.items():
        configs[key[:3]].append((key[3], mealy))

    sorted_configs = sorted(configs.items(), key=lambda x: x[0], reverse=True)

    for (s, a, p), machines_in_config in sorted_configs:
        config_timed_out = False
        for m_id, mealy in sorted(machines_in_config, key=lambda x: x[0]):
            print(f"S={s} A={a} U={p}% M={m_id}:", end=" ", flush=True)
            
            if config_timed_out:
                print("SKIPPED (Config Timeout)")
                res = {
                    'success': False, 'timeout': True, 'error': 'Config timeout exceeded',
                    'output_queries': '', 'equivalence_queries': '', 'sul_steps': '',
                    'total_time': timeout, 'learned_states': ''
                }
            else:
                res = run_safe_benchmark(mealy, 'lsharp_square', timeout=timeout)
                
                if res.get('timeout'):
                    print("TIMEOUT - skipping remaining machines in this config")
                    config_timed_out = True
                elif res['success']:
                    print(f"Completed (OQ: {res['output_queries']})")
                else:
                    print(f"X ERROR: {res.get('error', 'Unknown error')}")

            if res['success']: status = 'ok'
            elif res.get('timeout'): status = 'timeout'
            else: status = 'error'

            raw_data.append({
                'num_states': s, 'alphabet_size': a, 'percent_unknown': p, 'machine_id': m_id,
                'queries_learning': res.get('output_queries', ''),
                'equivalence_queries': res.get('equivalence_queries', ''),
                'sul_steps': res.get('sul_steps', ''),
                'total_time': res.get('total_time', ''),
                'learned_states': res.get('learned_states', ''),
                'bisimilar': res.get('success', ''),
                'success': res.get('success', False),
                'status': status
            })

    if raw_data:
        df = pd.DataFrame(raw_data)
        df.to_csv(results_dir / "unknowns_raw.csv", index=False)

        # Save averaged results
        metrics = ['queries_learning', 'equivalence_queries', 'sul_steps', 'total_time', 'learned_states']
        for m in metrics:
            df[m] = pd.to_numeric(df[m], errors='coerce')

        avg = df.groupby(['num_states', 'alphabet_size', 'percent_unknown'])[metrics].mean().reset_index()
        avg.columns = [f"avg_{c}" if c in metrics else c for c in avg.columns]

        counts = df.groupby(['num_states', 'alphabet_size', 'percent_unknown'])['success'].agg(['sum', 'count']).reset_index()
        avg['success_count'] = counts['sum']
        avg['total_count'] = counts['count']
        avg.to_csv(results_dir / "unknowns_averaged.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sequential Random Mealy Benchmarking")
    parser.add_argument('--mode', choices=['comparison', 'unknowns'], default='comparison')
    args = parser.parse_args()
    
    if args.mode == 'comparison':
        run_comparison_benchmark()
    elif args.mode == 'unknowns':
        run_unknown_benchmark()