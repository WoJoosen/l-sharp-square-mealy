"""
Benchmark L#Square algorithm against AALpy's L# algorithm
Compare both algorithms on the same test models
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aalpy.utils import load_automaton_from_file, bisimilar
from aalpy.oracles import PerfectKnowledgeEqOracle
from aalpy.learning_algs import run_Lsharp
from LSharpSquare.IncompleteMealySUL import MealySUL
from LSharpSquare.LSharpSquare import run_lsharp_square
from LSharpSquare.IncompleteKnowledgeEqOracle import IncompleteKnowledgeEqOracle
import pandas as pd

def get_models_up_to_n_states(max_states=100, models_dir=None):
    """Load all models with at most max_states states"""
    models = []
    if models_dir is None:
        models_dir = Path(__file__).parent / "models"
    
    for dot_file in sorted(models_dir.glob("*.dot")):
        try:
            mealy = load_automaton_from_file(str(dot_file), "mealy")
            if len(mealy.states) <= max_states:
                models.append({
                    'path': str(dot_file),
                    'name': dot_file.stem,
                    'states': len(mealy.states),
                    'machine': mealy
                })
        except Exception as e:
            print(f"Warning: Could not load {dot_file}: {e}")
    
    return sorted(models, key=lambda m: m['states'])

def benchmark_lsharp_square(model_path, timeout=120):
    """Benchmark L#Square"""
    
    print(f"    L#Square...", end=" ", flush=True)
    
    try:
        # Fresh machine for each test
        mealy = load_automaton_from_file(model_path, "mealy")
        alphabet = list(mealy.initial_state.transitions.keys())
        
        sul = MealySUL(mealy)
        eq_oracle = IncompleteKnowledgeEqOracle(alphabet, sul, mealy)
        
        learned_mealy, info = run_lsharp_square(
            alphabet, sul, eq_oracle, 
            return_data=True, 
            solver_timeout=timeout
        )
        
        # Check bisimilarity
        is_bisimilar = bisimilar(learned_mealy, mealy)
        
        result = {
            'algorithm': 'L#Square',
            'total_time': info['total_time'],
            'learning_time': info['learning_time'],
            'smt_time': info['smt_time'],
            'output_queries': info['queries_learning'],
            'equivalence_queries': info['queries_eq_oracle'],
            'learning_rounds': info['learning_rounds'],
            'sul_steps': info.get('sul_steps', 0),
            'bisimilar': is_bisimilar,
            'success': is_bisimilar
        }
        
        bisimilar_str = "Y" if is_bisimilar else "N"
        print(f"{info['queries_learning']} queries, {info['learning_rounds']} rounds, {info['total_time']:.3f}s {bisimilar_str}")
        return result
        
    except Exception as e:
        print(f"{str(e)[:40]}")
        return {
            'algorithm': 'L#Square',
            'success': False,
            'bisimilar': False,
            'error': str(e)[:50]
        }

def benchmark_aalpy_lsharp(model_path, timeout=120):
    """Benchmark AALpy's L# algorithm"""
    
    print(f"    AALpy L#...", end=" ", flush=True)
    
    try:
        mealy = load_automaton_from_file(model_path, "mealy")
        alphabet = list(mealy.initial_state.transitions.keys())
        
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

        sul = AalpySUL(mealy)
        eq_oracle = PerfectKnowledgeEqOracle(alphabet, sul, mealy)
        
        learned_mealy, info = run_Lsharp(
            alphabet=alphabet,
            sul=sul,
            eq_oracle=eq_oracle,
            automaton_type="mealy",
            return_data=True,
            print_level=1,
        )
        
        # Check bisimilarity
        is_bisimilar = bisimilar(learned_mealy, mealy)
        
        result = {
            'algorithm': 'AALpy L#',
            'total_time': info['total_time'],
            'learning_time': info['learning_time'],
            'output_queries': info['queries_learning'],
            'equivalence_queries': info['queries_eq_oracle'],
            'learning_rounds': info['learning_rounds'],
            'sul_steps': info.get('steps_learning', 0),
            'bisimilar': is_bisimilar,
            'success': is_bisimilar
        }
        
        bisimilar_str = "Y" if is_bisimilar else "N"
        print(f"{info['queries_learning']} queries, {info['learning_rounds']} rounds, {info['total_time']:.3f}s {bisimilar_str}")
        return result
        
    except Exception as e:
        print(f"{str(e)[:40]}")
        return {
            'algorithm': 'AALpy L#',
            'success': False,
            'bisimilar': False,
            'error': str(e)[:50]
        }

def run_protocols_benchmark(max_states=100, timeout=1000):
    """Benchmark L#Square vs AALpy on protocol models."""
    # Models dir
    models_dir = Path(__file__).parent.parent.parent / "models" / "protocols"
    models_dir.mkdir(exist_ok=True)

    # Create results directory
    results_dir = Path(__file__).parent.parent / "results" / "protocols"
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("L#Square Benchmark vs AALpy L#")
    print("="*80)
    print(f"Results will be saved to: {results_dir}")
    
    # Get models
    models = get_models_up_to_n_states(max_states, models_dir)
    
    print(f"\nFound {len(models)} models with ≤ {max_states} states:")
    for m in models:
        print(f"  - {m['name']}: {m['states']} states")
    
    all_results = {}
    
    # Benchmark each model
    for model_info in models:
        print(f"\n{'='*80}")
        print(f"Model: {model_info['name']} ({model_info['states']} states)")
        print('='*80)
        
        # Benchmark L#Square with k=1
        print("\nBenchmarking algorithms:")
        lsharp_result = benchmark_lsharp_square(model_info['path'], timeout=timeout)
        
        # Benchmark AALpy's L#
        aalpy_result = benchmark_aalpy_lsharp(model_info['path'], timeout=timeout)
        
        all_results[model_info['name']] = {
            'states': model_info['states'],
            'name': model_info['name'],
            'lsharp': lsharp_result,
            'aalpy': aalpy_result
        }
    
    # Summary and comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY & COMPARISON")
    print('='*80)
    
    summary_data = []
    
    for model_name in sorted(all_results.keys(), key=lambda m: all_results[m]['states']):
        model_results = all_results[model_name]
        print(f"\n{model_name} ({model_results['states']} states):")
        print("-" * 100)
        
        lsharp_result = model_results['lsharp']
        aalpy_result = model_results['aalpy']
        
        # L#Square results
        if lsharp_result['success']:
            print(f"  L#Square:")
            print(f"    Rounds: {lsharp_result['learning_rounds']}")
            print(f"    Total Time: {lsharp_result['total_time']:.3f}s (Learning: {lsharp_result['learning_time']:.3f}s, SMT: {lsharp_result['smt_time']:.3f}s)")
            print(f"    Output Queries: {lsharp_result['output_queries']}")
            print(f"    Equivalence Queries: {lsharp_result['equivalence_queries']}")
            print(f"    SUL Steps: {lsharp_result['sul_steps']}")
            print(f"    Bisimilar: {'YES' if lsharp_result.get('bisimilar', False) else 'NO'}")
            
            summary_data.append({
                'Model': model_name,
                'States': model_results['states'],
                'Algorithm': 'L#Square',
                'Rounds': lsharp_result['learning_rounds'],
                'Total Time (s)': f"{lsharp_result['total_time']:.3f}",
                'Output Queries': lsharp_result['output_queries'],
                'Equivalence Queries': lsharp_result['equivalence_queries'],
                'SUL Steps': lsharp_result['sul_steps'],
                'Bisimilar': 'Yes' if lsharp_result.get('bisimilar', False) else 'No'
            })
        else:
            print(f"  L#Square: FAILED - {lsharp_result.get('error', 'Unknown error')}")
        
        # AALpy results
        if aalpy_result['success']:
            print(f"  AALpy L#:")
            print(f"    Rounds: {aalpy_result['learning_rounds']}")
            print(f"    Total Time: {aalpy_result['total_time']:.3f}s (Learning: {aalpy_result['learning_time']:.3f}s)")
            print(f"    Output Queries: {aalpy_result['output_queries']}")
            print(f"    Equivalence Queries: {aalpy_result['equivalence_queries']}")
            print(f"    SUL Steps: {aalpy_result['sul_steps']}")
            print(f"    Bisimilar: {'YES' if aalpy_result.get('bisimilar', False) else 'NO'}")
            
            summary_data.append({
                'Model': model_name,
                'States': model_results['states'],
                'Algorithm': 'AALpy L#',
                'Rounds': aalpy_result['learning_rounds'],
                'Total Time (s)': f"{aalpy_result['total_time']:.3f}",
                'Output Queries': aalpy_result['output_queries'],
                'Equivalence Queries': aalpy_result['equivalence_queries'],
                'SUL Steps': aalpy_result['sul_steps'],
                'Bisimilar': 'Yes' if aalpy_result.get('bisimilar', False) else 'No'
            })
        else:
            print(f"  AALpy L#: FAILED - {aalpy_result.get('error', 'Unknown error')}")
        
        # Comparison
        if lsharp_result['success'] and aalpy_result['success']:
            print(f"\n  Comparison:")
            time_ratio = lsharp_result['total_time'] / aalpy_result['total_time'] if aalpy_result['total_time'] > 0 else 0
            query_ratio = lsharp_result['output_queries'] / aalpy_result['output_queries'] if aalpy_result['output_queries'] > 0 else 0
            rounds_ratio = lsharp_result['learning_rounds'] / aalpy_result['learning_rounds'] if aalpy_result['learning_rounds'] > 0 else 0
            sul_steps_ratio = lsharp_result['sul_steps'] / aalpy_result['sul_steps'] if aalpy_result['sul_steps'] > 0 else 0
            if time_ratio > 0:
                print(f"    Time: L#Square is {time_ratio:.2f}x {'faster' if time_ratio < 1 else 'slower'}")
            if query_ratio > 0:
                print(f"    Queries: L#Square uses {query_ratio:.2f}x {'fewer' if query_ratio < 1 else 'more'} output queries")
            if rounds_ratio > 0:
                print(f"    Rounds: L#Square needs {rounds_ratio:.2f}x {'fewer' if rounds_ratio < 1 else 'more'} learning rounds")
            if sul_steps_ratio > 0:
                print(f"    SUL Steps: L#Square uses {sul_steps_ratio:.2f}x {'fewer' if sul_steps_ratio < 1 else 'more'} input symbols")
    
    # Save summary to CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        csv_path = results_dir / 'benchmark_results.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSummary saved to {csv_path.name}")
    
    # Check for non-bisimilar results
    non_bisimilar_tests = []
    for model_name in sorted(all_results.keys(), key=lambda m: all_results[m]['states']):
        model_results = all_results[model_name]
        lsharp_result = model_results['lsharp']
        aalpy_result = model_results['aalpy']
        
        if lsharp_result.get('success') and not lsharp_result.get('bisimilar', False):
            non_bisimilar_tests.append((model_name, 'L#Square', model_results['states']))
        
        if aalpy_result.get('success') and not aalpy_result.get('bisimilar', False):
            non_bisimilar_tests.append((model_name, 'AALpy L#', model_results['states']))
    
    print(f"\n{'='*80}")
    print("BISIMILARITY CHECK RESULTS")
    print('='*80)
    
    if non_bisimilar_tests:
        print(f"\nWARNING: {len(non_bisimilar_tests)} test(s) resulted in non-bisimilar machines:\n")
        for model_name, algorithm, num_states in non_bisimilar_tests:
            print(f"  X {model_name} ({num_states} states) - {algorithm}")
    else:
        print("\nAll tests successfully produced bisimilar machines!")
    
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results saved to: {results_dir}")
    print('='*80)

if __name__ == "__main__":
    run_protocols_benchmark()
