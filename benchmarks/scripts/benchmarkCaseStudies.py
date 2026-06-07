"""
Unified Case Study Benchmark: Adder, Boolean Stack, and Loop Machines.
Runs L#Square on various structural models with unknown outputs.
"""

import csv
import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Ensure files can be imported from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aalpy.utils import load_automaton_from_file
from LSharpSquare.IncompleteKnowledgeEqOracle import IncompleteKnowledgeEqOracle, bisimilar_with_unknowns
from LSharpSquare.IncompleteMealySUL import IncompleteMealySUL
from LSharpSquare.LSharpSquare import run_lsharp_square
from models.generators import (generate_adder_mealy_dot, 
                               generate_boolean_stack_dot, 
                               generate_loop_mealy_dot)


def benchmark_adder_machines(n_values: list, model_dir: Path, timeout: int = 120) -> Dict:
    """Benchmark adder machines where n is the max sum/alphabet size."""
    metrics = {}
    total_tests = len(n_values)
    print(f"\n--- Benchmarking Adder Machines: {total_tests} configurations ---")
    
    for i, n in enumerate(n_values, 1):
        print(f"[{i}/{total_tests}] Testing n={n}...", end=" ", flush=True)
        try:
            generate_adder_mealy_dot(n, model_dir)
            model_path = model_dir / f"adder_max{n}.dot"
            target_mealy = load_automaton_from_file(str(model_path), "mealy")
            alphabet = list(target_mealy.initial_state.transitions.keys())
            
            sul = IncompleteMealySUL([], target_mealy)
            eq_oracle = IncompleteKnowledgeEqOracle(alphabet, sul, target_mealy)
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_lsharp_square, alphabet, sul, eq_oracle, return_data=True)
                learned_mealy, info = future.result(timeout=timeout)
            elapsed_time = time.time() - start_time
            
            if bisimilar_with_unknowns(learned_mealy, target_mealy):
                metrics[n] = {
                    'n': n, 'target_states': len(target_mealy.states), 'learned_states': len(learned_mealy.states),
                    'alphabet_size': len(alphabet), 'output_queries': info.get('queries_learning', 0),
                    'equivalence_queries': info.get('validity_query', 0), 'time_s': info.get('total_time', elapsed_time)
                }
                print(f"OK ({metrics[n]['time_s']:.2f}s)")
            else:
                print("FAIL (Not Bisimilar)")
        except concurrent.futures.TimeoutError:
            print(f"TIMEOUT ({timeout}s exceeded)")
        except Exception as e:
            print(f"ERROR: {e}")
    return metrics


def benchmark_boolean_stacks(n_values: list, model_dir: Path, timeout: int = 120) -> Dict:
    """Benchmark boolean stacks where n is the depth."""
    metrics = {}
    total_tests = len(n_values)
    print(f"\n--- Benchmarking Boolean Stacks: {total_tests} depths ---")
    
    for i, n in enumerate(n_values, 1):
        print(f"[{i}/{total_tests}] Testing depth n={n}...", end=" ", flush=True)
        try:
            generate_boolean_stack_dot(n, model_dir)
            model_path = model_dir / f"boolean_stack_d{n}.dot"
            target_mealy = load_automaton_from_file(str(model_path), "mealy")
            alphabet = list(target_mealy.initial_state.transitions.keys())
            
            sul = IncompleteMealySUL([], target_mealy)
            eq_oracle = IncompleteKnowledgeEqOracle(alphabet, sul, target_mealy)
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_lsharp_square, alphabet, sul, eq_oracle, return_data=True)
                learned_mealy, info = future.result(timeout=timeout)
            elapsed_time = time.time() - start_time
            
            if bisimilar_with_unknowns(learned_mealy, target_mealy):
                metrics[n] = {
                    'n': n, 'target_states': len(target_mealy.states), 'learned_states': len(learned_mealy.states),
                    'output_queries': info.get('queries_learning', 0), 'equivalence_queries': info.get('validity_query', 0),
                    'time_s': info.get('total_time', elapsed_time)
                }
                print(f"OK ({metrics[n]['time_s']:.2f}s)")
            else:
                print("FAIL (Not Bisimilar)")
        except concurrent.futures.TimeoutError:
            print(f"TIMEOUT ({timeout}s exceeded)")
        except Exception as e:
            print(f"ERROR: {e}")
    return metrics


def benchmark_loop_machines(n_values: list, k_values: list, model_dir: Path, timeout: int = 120) -> Tuple[Dict, Dict]:
    """Benchmark loop machines with size N and k unknowns."""
    results, metrics = {}, {}
    total_tests = len(n_values) * len(k_values)
    test_num = 0
    print(f"\n--- Benchmarking Loop Machines: {len(n_values)} sizes × {len(k_values)} unknown counts ---")
    
    for N in n_values:
        for k in k_values:
            test_num += 1
            if k >= N:
                continue
            
            print(f"[{test_num}/{total_tests}] Testing N={N}, k={k}...", end=" ", flush=True)
            try:
                generate_loop_mealy_dot(N, k, model_dir)
                model_path = model_dir / f"loop_s{N}_u{k}.dot"
                target_mealy = load_automaton_from_file(str(model_path), "mealy")
                alphabet = list(target_mealy.initial_state.transitions.keys())
                
                sul = IncompleteMealySUL([], target_mealy)
                eq_oracle = IncompleteKnowledgeEqOracle(alphabet, sul, target_mealy)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_lsharp_square, alphabet, sul, eq_oracle, return_data=True)
                    learned_mealy, info = future.result(timeout=timeout)
                
                if bisimilar_with_unknowns(learned_mealy, target_mealy):
                    results[(N, k)] = len(learned_mealy.states)
                    metrics[(N, k)] = {
                        'output_queries': info.get('queries_learning', 0),
                        'equivalence_queries': info.get('validity_query', 0),
                        'sul_steps': info.get('sul_steps', 0),
                        'time_s': info.get('total_time', 0),
                        'learned_states': len(learned_mealy.states),
                        'target_states': len(target_mealy.states),
                        'status': 'ok'
                    }
                    print(f"OK ({metrics[(N, k)]['time_s']:.2f}s)")
            except concurrent.futures.TimeoutError:
                print(f"TIMEOUT ({timeout}s exceeded)")
            except Exception as e:
                print(f"ERROR: {e}")
    return results, metrics


def save_adder_csv(metrics, n_values, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n (max sum)', 'target_states', 'learned_states', 'alphabet_size', 'output_queries', 'equivalence_queries', 'time_s'])
        for n in n_values:
            if n in metrics:
                m = metrics[n]
                writer.writerow([m['n'], m['target_states'], m['learned_states'], m['alphabet_size'], m['output_queries'], m['equivalence_queries'], m['time_s']])


def save_boolean_csv(metrics, n_values, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n (depth)', 'target_states', 'learned_states', 'output_queries', 'equivalence_queries', 'time_s'])
        for n in n_values:
            if n in metrics:
                m = metrics[n]
                writer.writerow([m['n'], m['target_states'], m['learned_states'], m['output_queries'], m['equivalence_queries'], m['time_s']])


def save_loop_metrics_csv(metrics, n_values, k_values, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'k', 'output_queries', 'equivalence_queries', 'sul_steps', 'time_s', 'learned_states', 'target_states', 'status'])
        for N in n_values:
            for k in k_values:
                if (N, k) in metrics:
                    m = metrics[(N, k)]
                    writer.writerow([N, k, m['output_queries'], m['equivalence_queries'], m['sul_steps'], m['time_s'], m['learned_states'], m['target_states'], m['status']])


def run_all_case_studies(timeout=120):
    """Run all case study benchmarks."""
    benchmark_dir = Path(__file__).parent.parent
    base_dir = benchmark_dir.parent
    results_dir = benchmark_dir / "results" / "caseStudies"
    model_dir = base_dir / "models" / "caseStudies"
    results_dir.mkdir(parents=True, exist_ok=True)

    max_adder_n = 20
    max_stack_n = 5
    max_loop_n_k = (18,16)

    print("=" * 80)
    print("L#Square Case Study Benchmarks")
    print("=" * 80)

    # 1. Adder Machines
    adder_n = list(range(2, max_adder_n + 1))
    adder_metrics = benchmark_adder_machines(adder_n, model_dir / "adders", timeout=timeout)
    save_adder_csv(adder_metrics, adder_n, results_dir / "case_study_adder.csv")

    # 2. Boolean Stacks
    stack_n = list(range(1, max_stack_n + 1))
    stack_metrics = benchmark_boolean_stacks(stack_n, model_dir / "stacks", timeout=timeout)
    save_boolean_csv(stack_metrics, stack_n, results_dir / "case_study_boolean_stack.csv")

    # 3. Loop Machines
    loop_n, loop_k = list(range(4, max_loop_n_k[0] + 1)), list(range(2, max_loop_n_k[1] + 1))
    _, loop_metrics = benchmark_loop_machines(loop_n, loop_k, model_dir / "loops", timeout=timeout)
    save_loop_metrics_csv(loop_metrics, loop_n, loop_k, results_dir / "case_study_loop.csv")

    print("\nAll case studies complete. Results saved to:", results_dir)

if __name__ == "__main__":
    run_all_case_studies()
