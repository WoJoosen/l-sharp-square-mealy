import argparse
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aalpy.utils import load_automaton_from_file
from LSharpSquare.IncompleteMealySUL import IncompleteMealySUL
from LSharpSquare.IncompleteKnowledgeEqOracle import IncompleteKnowledgeEqOracle, bisimilar_with_unknowns
from LSharpSquare.LSharpSquare import run_lsharp_square
from benchmarks.scripts.benchmarkRandomScaling import run_comparison_benchmark, run_unknown_benchmark
from benchmarks.scripts.benchmarkCaseStudies import run_all_case_studies
from benchmarks.scripts.benchmarkProtocols import run_protocols_benchmark
from benchmarks.scripts.plotResults import run_all_plots

def run_single_model(model_path, timeout=120):
    """Loads and learns a single Mealy machine."""

    loaded_mealy = load_automaton_from_file(model_path, "mealy")
    alphabet = list(loaded_mealy.initial_state.transitions.keys())

    sul = IncompleteMealySUL([], loaded_mealy)
    eq_oracle = IncompleteKnowledgeEqOracle(alphabet, sul, loaded_mealy)
    
    learned_mealy, info = run_lsharp_square(alphabet, sul, eq_oracle, return_data=True, solver_timeout=timeout)

    print(f"\nBisimilar: {bisimilar_with_unknowns(learned_mealy, loaded_mealy)}")
    print(f"Total time: {info.get('total_time', 'N/A')}s")
    print("Info:", info)

def main():
    parser = argparse.ArgumentParser(description="L#Square Mealy CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run command: Execute a single model
    run_parser = subparsers.add_parser("run", help="Run L#Square on a specific .dot model")
    run_parser.add_argument("model", type=str, help="Path to the .dot file")

    # Benchmark command: Run a suite
    bench_parser = subparsers.add_parser("benchmark", help="Run automated benchmarks")
    bench_parser.add_argument("type", choices=["random-comparison", "random-unknowns", "case-studies", "protocols"],
                              help="The benchmark category to run")
    bench_parser.add_argument("--timeout", type=int, default=120, 
                              help="Timeout (seconds) for each learning task (default: 120)")

    # Plot command: Generate visualizations
    plot_parser = subparsers.add_parser("plot", help="Generate plots from current benchmark results")

    args = parser.parse_args()

    if args.command == "run":
        print(f"Executing L#Square on {args.model}...")
        run_single_model(args.model)
    
    elif args.command == "benchmark":
        if args.type == "random-comparison":
            run_comparison_benchmark(timeout=args.timeout)
        elif args.type == "random-unknowns":
            run_unknown_benchmark(timeout=args.timeout)
        elif args.type == "case-studies":
            run_all_case_studies(timeout=args.timeout)
        elif args.type == "protocols":
            run_protocols_benchmark(timeout=args.timeout)
            
    elif args.command == "plot":
        run_all_plots()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
