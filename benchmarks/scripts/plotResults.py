"""
Unified plotting script for L#Square benchmark results.
Combines:
1. 2x2 Bar Chart (K1 benchmarks)
2. 4x4 Grid Comparison (Random machines)
3. 5x4 Grid for Unknown Outputs (Random machines with unknowns)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_k1_2x2(csv_path, output_dir):
    """Create 2x2 bar chart subplots for OQ, EQ, SUL Steps, Time (from plot_results_k1.py)"""
    if not csv_path.exists():
        print(f"Skipping 2x2: {csv_path} not found.")
        return

    print(f"Generating 2x2 K1 Comparison from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    all_results = {}
    for _, row in df.iterrows():
        model_name = row['Model']
        if model_name not in all_results:
            all_results[model_name] = {'states': row['States'], 'name': model_name, 'lsharp': {}, 'aalpy': {}}
        
        algorithm = row['Algorithm']
        result_dict = {
            'learning_rounds': row['Rounds'],
            'total_time': float(row['Total Time (s)']),
            'output_queries': int(row['Output Queries']),
            'equivalence_queries': int(row['Equivalence Queries']),
            'sul_steps': int(row['SUL Steps']),
            'success': True
        }
        if algorithm == 'L#Square':
            all_results[model_name]['lsharp'] = result_dict
        else:
            all_results[model_name]['aalpy'] = result_dict

    models_list = sorted(all_results.keys(), key=lambda m: all_results[m]['states'])
    model_names_short = [all_results[m]['name'][:12] for m in models_list]
    x_pos = np.arange(len(models_list))
    width = 0.35
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ('output_queries', 'Output Queries', axes[0, 0]),
        ('equivalence_queries', 'Equivalence Queries', axes[0, 1]),
        ('sul_steps', 'SUL Steps', axes[1, 0]),
        ('total_time', 'Execution Time (s)', axes[1, 1])
    ]
    
    for metric_key, metric_label, ax in metrics:
        lsharp_vals = [all_results[m]['lsharp'].get(metric_key, 0) if all_results[m]['lsharp'].get('success', False) else 0 for m in models_list]
        aalpy_vals = [all_results[m]['aalpy'].get(metric_key, 0) if all_results[m]['aalpy'].get('success', False) else 0 for m in models_list]
        
        bars1 = ax.bar(x_pos - width/2, lsharp_vals, width, label='L#Square', color='#1f77b4', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, aalpy_vals, width, label='AALpy L#', color='#ff7f0e', alpha=0.8, edgecolor='black')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if height < 0.05:
                        label = '<0.1'
                    elif height == int(height):
                        label = f'{int(height)}'
                    else:
                        label = f'{height:.1f}'
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=8)

        ax.set_ylabel(metric_label, fontsize=15, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names_short, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, which='both')
        if metric_key == 'total_time':
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison_2x2.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_random_4x4(csv_path, output_dir):
    """Create 4x4 grid: Rows=Metrics, Columns=Alphabet Sizes (from plot_results.py)"""
    if not csv_path.exists():
        print(f"Skipping 4x4: {csv_path} not found.")
        return

    print(f"Generating 4x4 Random Comparison from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    alphabet_sizes = [2, 4, 8, 16]
    state_counts = sorted(df['States'].unique())
    colors = {'L#Square': '#1f77b4', 'AALpy': '#ff7f0e'}
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    metrics = [('OQ', 'Output Queries'), ('EQ', 'Equivalence Queries'), ('Steps', 'Total SUL Steps'), ('Time', 'Execution Time (s)')]
    
    for row, (metric_key, metric_label) in enumerate(metrics):
        for col, alphabet_size in enumerate(alphabet_sizes):
            ax = axes[row, col]
            data = df[df['Alphabet'] == alphabet_size].sort_values('States')
            x = data['States'].values
            
            for alg, label, marker in [('L#Square', 'L#Square', 's'), ('AALpy', 'AALpy L#', 'o')]:
                y_mean = data[f'{alg}_{metric_key}_Mean'].values
                y_std = data[f'{alg}_{metric_key}_Std'].values
                ax.plot(x, y_mean, marker=marker, color=colors[alg], label=label, linewidth=2.5, markersize=6)
                ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=colors[alg], alpha=0.2)

            if row == 3: ax.set_xlabel('States', fontsize=15, fontweight='bold')
            if col == 0: ax.set_ylabel(metric_label, fontsize=15, fontweight='bold')
            if row == 0: ax.set_title(f'|Σ| = {alphabet_size}', fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(state_counts)
            ax.tick_params(axis='both', labelsize=12)
            if metric_key == 'Time': ax.set_yscale('log')
            if row == 0 and col == 0: ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_visualization_4x4.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_unknowns_5x4(csv_path, output_dir):
    """Create 5x4 grid for unknown percentages (from plot_results_unknowns.py)"""
    if not csv_path.exists():
        print(f"Skipping 5x4: {csv_path} not found.")
        return

    print(f"Generating 5x4 Unknown Grid from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    metrics = [
        ('avg_queries_learning', 'Output Queries'),
        ('avg_equivalence_queries', 'Equivalence Queries'),
        ('avg_sul_steps', 'SUL Steps'),
        ('avg_total_time', 'Total Time (s)'),
        ('avg_learned_states', 'Learned States')
    ]
    alphabet_sizes = [2, 4, 8, 16]
    unknown_percentages = [0, 20, 40, 60]
    state_sizes = sorted(df['num_states'].unique())
    
    fig, axes = plt.subplots(5, 4, figsize=(16, 16))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    
    for row_idx, (metric_key, metric_label) in enumerate(metrics):
        for col_idx, alphabet_size in enumerate(alphabet_sizes):
            ax = axes[row_idx, col_idx]
            for percent_idx, percent_unknown in enumerate(unknown_percentages):
                filtered = df[(df['alphabet_size'] == alphabet_size) & 
                             (df['percent_unknown'] == percent_unknown) &
                             (df['success_count'] == df['total_count'])].sort_values('num_states')
                
                if not filtered.empty:
                    x_data = filtered['num_states'].values
                    y_data = pd.to_numeric(filtered[metric_key], errors='coerce').values
                    ax.plot(x_data, y_data, marker=markers[percent_idx], linewidth=2, markersize=6,
                           color=colors[percent_idx], linestyle=line_styles[percent_idx],
                           label=f'{percent_unknown}% unknown', alpha=0.8)
            
            if col_idx == 0: ax.set_ylabel(metric_label, fontsize=15, fontweight='bold')
            if row_idx == 4: ax.set_xlabel('Number of States', fontsize=15, fontweight='bold')
            else: ax.set_xticklabels([])
            ax.set_xticks(state_sizes)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, alpha=0.3, which='both')
            if metric_key == 'avg_total_time': ax.set_yscale('log')
            if row_idx == 0: ax.set_title(f'|Σ| = {alphabet_size}', fontsize=15, fontweight='bold')
            if row_idx == 0 and col_idx == 0: ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / 'benchmark_unknowns_5x4.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_all_plots():
    # Define paths relative to the root or current script
    base_path = Path(__file__).parent.parent
    
    # 1. K1 Results (2x2)
    k1_csv = base_path / "results" / "protocols" / "benchmark_results.csv"
    k1_out = base_path / "plots" / "protocols"
    k1_out.mkdir(parents=True, exist_ok=True)
    plot_k1_2x2(k1_csv, k1_out)
    
    # 2. Random Comparison (4x4)
    random_csv = base_path / "results" / "randomScaling" / "comparison_averaged.csv"
    random_out = base_path / "plots" / "randomScaling"
    random_out.mkdir(parents=True, exist_ok=True)
    plot_random_4x4(random_csv, random_out)
    
    # 3. Unknowns (5x4)
    unknown_csv = base_path / "results" / "randomScalingUnknown" / "unknowns_averaged.csv"
    unknown_out = base_path / "plots" / "randomScalingUnknown"
    unknown_out.mkdir(parents=True, exist_ok=True)
    plot_unknowns_5x4(unknown_csv, unknown_out)
    
    print("\nAll requested visualizations have been generated.")

if __name__ == "__main__":
    run_all_plots()
