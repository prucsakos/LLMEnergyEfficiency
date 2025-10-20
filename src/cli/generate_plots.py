#!/usr/bin/env python3
"""
Enhanced CLI script for generating high-quality plots from wandb data.

Generates 7 types of plots:
1. Accuracy vs Latency (original)
2. Accuracy vs Generated Tokens (original)
3. Pareto: Accuracy vs Energy
4. Scatter: Accuracy vs Release Date (with trend line)
5. Pareto: Accuracy vs Tokens per Correct
6. Pareto: Tokens per Correct vs Latency
7. Leaderboards: Accuracy and Tokens per Correct
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy import stats
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def setup_plotting_style():
    """Set up high-quality plotting style."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['legend.fontsize'] = 8

def query_wandb_data(project_name: str) -> pd.DataFrame:
    """Query wandb data for a given project and return as DataFrame."""
    print(f"Querying wandb data for project: {project_name}")
    
    api = wandb.Api()
    runs = api.runs(project_name)
    
    data = []
    for run in runs:
        try:
            config = run.config
            summary = run.summary
            
            model_name = config.get('model_name', config.get('model', 'unknown'))
            if '/' in model_name:
                model_name = model_name.split('/', 1)[1]
            
            run_data = {
                'run_id': run.id,
                'run_name': run.name,
                'model_family': config.get('model_family', 'unknown'),
                'model_name': model_name,
                'model_repo': config.get('model', 'unknown'),
                'dataset': config.get('dataset', 'unknown'),
                'latency_ms': summary.get('performance_avg_latency_ms', None),
                'self_eval_acc': summary.get('evaluation_self_eval_accuracy', None),
                'avg_gen_tokens': summary.get('tokens_avg_generated', summary.get('avg_gen_tokens', None)),
                'think_budget': config.get('think_budget', None),
                'style': config.get('style', None),
                'K': config.get('K', None),
                'batch_size': summary.get('batch_size', None),
                'params_B': summary.get('model_parameters_billions', None),
                'release_date': summary.get('model_release_date', None),
                'tokens_per_correct': summary.get('efficiency_tokens_per_correct_mean', None),
                'energy_joules': summary.get('energy_joules_per_correct_answer', None),
            }
            
            if (run_data['latency_ms'] is not None and 
                run_data['self_eval_acc'] is not None and 
                run_data['avg_gen_tokens'] is not None and
                run.state == 'finished'):
                data.append(run_data)
                
        except Exception as e:
            print(f"Warning: Skipping run {run.id} due to error: {e}")
            continue
    
    if not data:
        raise ValueError(f"No valid runs found in project {project_name}")
    
    df = pd.DataFrame(data)
    print(f"Retrieved {len(df)} valid runs")
    
    return df

def calculate_pareto_frontier(x: np.ndarray, y: np.ndarray, 
                            minimize_x: bool = True, minimize_y: bool = False) -> np.ndarray:
    """Calculate pareto frontier points. Minimize x and maximize y by default."""
    if len(x) == 0:
        return np.array([])
    
    pareto_indices = []
    
    for i in range(len(x)):
        is_pareto = True
        for j in range(len(x)):
            if i != j:
                if (x[j] <= x[i] and y[j] >= y[i]) and (x[j] < x[i] or y[j] > y[i]):
                    is_pareto = False
                    break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return np.array(pareto_indices)

def plot_accuracy_vs_energy(df: pd.DataFrame, output_dir: Path, pareto: bool = True):
    """Plot 1: Pareto with energy on x-axis, accuracy on y-axis."""
    print("Creating accuracy vs energy plots (Pareto)...")
    
    plot_df = df.dropna(subset=['self_eval_acc', 'energy_joules', 'params_B', 'avg_gen_tokens']).copy()
    
    if len(plot_df) == 0:
        print("Warning: No data available for energy plots")
        return
    
    datasets = sorted(plot_df['dataset'].unique())
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    if n_datasets == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        dataset_df = plot_df[plot_df['dataset'] == dataset]
        
        if len(dataset_df) == 0:
            ax.set_visible(False)
            continue
        
        scatter = ax.scatter(
            dataset_df['energy_joules'],
            dataset_df['self_eval_acc'],
            c=dataset_df['params_B'],
            s=50 + (dataset_df['avg_gen_tokens'] / 20000) * 150,
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        for _, row in dataset_df.iterrows():
            ax.annotate(row['model_name'], 
                       xy=(row['energy_joules'], row['self_eval_acc']),
                       xytext=(0, -8), textcoords='offset points',
                       fontsize=5, ha='center', va='top')
            ax.annotate(f"{int(row['avg_gen_tokens'])}", 
                       xy=(row['energy_joules'], row['self_eval_acc']),
                       xytext=(0, 8), textcoords='offset points',
                       fontsize=5, ha='center', va='bottom')
        
        if pareto and len(dataset_df) > 1:
            x_vals = dataset_df['energy_joules'].values
            y_vals = dataset_df['self_eval_acc'].values
            
            pareto_indices = calculate_pareto_frontier(x_vals, y_vals, minimize_x=True, minimize_y=False)
            
            if len(pareto_indices) > 1:
                pareto_x = x_vals[pareto_indices]
                pareto_y = y_vals[pareto_indices]
                
                sort_idx = np.argsort(pareto_x)
                ax.plot(pareto_x[sort_idx], pareto_y[sort_idx], 'r--', linewidth=2, alpha=0.8, label='Pareto Frontier')
        
        ax.set_xlabel('Energy (Joules per Correct)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset}')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Model Size (B)', fontsize=8)
    
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Accuracy vs Energy (Color: Model Size, Size: Tokens Generated)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_accuracy_vs_energy.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'pareto_accuracy_vs_energy.png'}")
    plt.close()

def plot_accuracy_vs_release_date(df: pd.DataFrame, output_dir: Path):
    """Plot 2: Scatter with release date on x-axis, accuracy on y-axis, with trend line."""
    print("Creating accuracy vs release date plots...")
    
    plot_df = df[df['release_date'].notna()].copy()
    plot_df = plot_df.dropna(subset=['self_eval_acc', 'params_B', 'avg_gen_tokens'])
    
    if len(plot_df) == 0:
        print("Warning: No data available for release date plots")
        return
    
    plot_df['release_date_parsed'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    plot_df = plot_df[plot_df['release_date_parsed'].notna()].copy()
    
    if len(plot_df) == 0:
        print("Warning: No valid release dates found")
        return
    
    datasets = sorted(plot_df['dataset'].unique())
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    if n_datasets == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        dataset_df = plot_df[plot_df['dataset'] == dataset]
        
        if len(dataset_df) == 0:
            ax.set_visible(False)
            continue
        
        scatter = ax.scatter(
            dataset_df['release_date_parsed'],
            dataset_df['self_eval_acc'],
            c=dataset_df['params_B'],
            s=50 + (dataset_df['avg_gen_tokens'] / 20000) * 150,
            cmap='viridis',
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add trend line
        x_numeric = pd.to_datetime(dataset_df['release_date_parsed']).astype(np.int64) / 10**9
        y_vals = dataset_df['self_eval_acc'].values
        if len(x_numeric) > 2:
            z = np.polyfit(x_numeric, y_vals, 1)
            p = np.poly1d(z)
            x_line_numeric = np.linspace(x_numeric.min(), x_numeric.max(), 100)
            x_line_dates = pd.to_datetime(x_line_numeric, unit='s')
            y_line = p(x_line_numeric)
            ax.plot(x_line_dates, y_line, 'r--', linewidth=2, alpha=0.8, label='Trend line')
        
        for _, row in dataset_df.iterrows():
            ax.annotate(row['model_name'], 
                       xy=(row['release_date_parsed'], row['self_eval_acc']),
                       xytext=(0, -8), textcoords='offset points',
                       fontsize=5, ha='center', va='top')
            ax.annotate(f"{int(row['avg_gen_tokens'])}", 
                       xy=(row['release_date_parsed'], row['self_eval_acc']),
                       xytext=(0, 8), textcoords='offset points',
                       fontsize=5, ha='center', va='bottom')
        
        ax.set_xlabel('Release Date')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset}')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Model Size (B)', fontsize=8)
    
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Accuracy vs Release Date (Color: Model Size, Size: Tokens Generated, Line: Trend)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_accuracy_vs_release_date.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'scatter_accuracy_vs_release_date.png'}")
    plt.close()

def plot_accuracy_vs_tokens_per_correct(df: pd.DataFrame, output_dir: Path, pareto: bool = True):
    """Plot 3: Pareto with tokens_per_correct on x-axis, accuracy on y-axis."""
    print("Creating accuracy vs tokens_per_correct plots (Pareto)...")
    
    plot_df = df.dropna(subset=['self_eval_acc', 'tokens_per_correct', 'params_B', 'avg_gen_tokens']).copy()
    
    if len(plot_df) == 0:
        print("Warning: No data available for tokens_per_correct plots")
        return
    
    datasets = sorted(plot_df['dataset'].unique())
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    if n_datasets == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        dataset_df = plot_df[plot_df['dataset'] == dataset]
        
        if len(dataset_df) == 0:
            ax.set_visible(False)
            continue
    
        scatter = ax.scatter(
            dataset_df['tokens_per_correct'],
            dataset_df['self_eval_acc'],
            c=dataset_df['params_B'],
            s=50 + (dataset_df['avg_gen_tokens'] / 20000) * 150,
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        for _, row in dataset_df.iterrows():
            ax.annotate(row['model_name'], 
                       xy=(row['tokens_per_correct'], row['self_eval_acc']),
                       xytext=(0, -8), textcoords='offset points',
                       fontsize=5, ha='center', va='top')
            ax.annotate(f"{int(row['avg_gen_tokens'])}", 
                       xy=(row['tokens_per_correct'], row['self_eval_acc']),
                       xytext=(0, 8), textcoords='offset points',
                       fontsize=5, ha='center', va='bottom')
        
        if pareto and len(dataset_df) > 1:
            x_vals = dataset_df['tokens_per_correct'].values
            y_vals = dataset_df['self_eval_acc'].values
            
            pareto_indices = calculate_pareto_frontier(x_vals, y_vals, minimize_x=True, minimize_y=False)
            
            if len(pareto_indices) > 1:
                pareto_x = x_vals[pareto_indices]
                pareto_y = y_vals[pareto_indices]
                
                sort_idx = np.argsort(pareto_x)
                ax.plot(pareto_x[sort_idx], pareto_y[sort_idx], 'r--', linewidth=2, alpha=0.8, label='Pareto Frontier')
        
        ax.set_xlabel('Tokens per Correct Answer')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset}')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Model Size (B)', fontsize=8)
    
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Accuracy vs Tokens per Correct (Color: Model Size, Size: Tokens Generated)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_accuracy_vs_tokens_per_correct.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'pareto_accuracy_vs_tokens_per_correct.png'}")
    plt.close()
    
def plot_efficiency_pareto(df: pd.DataFrame, output_dir: Path, pareto: bool = True):
    """Plot 4 & 5: Pareto with tokens_per_correct on x-axis, latency/tokens on y-axis."""
    print("Creating tokens_per_correct vs latency plots (Pareto)...")
    
    plot_df = df.dropna(subset=['latency_ms', 'tokens_per_correct', 'params_B', 'avg_gen_tokens']).copy()
    
    if len(plot_df) == 0:
        print("Warning: No data available for efficiency pareto plots")
        return

    datasets = sorted(plot_df['dataset'].unique())
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    if n_datasets == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        dataset_df = plot_df[plot_df['dataset'] == dataset]
        
        if len(dataset_df) == 0:
            ax.set_visible(False)
            continue
        
        scatter = ax.scatter(
            dataset_df['tokens_per_correct'],
            dataset_df['latency_ms'],
            c=dataset_df['params_B'],
            s=50 + (dataset_df['avg_gen_tokens'] / 20000) * 150,
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        for _, row in dataset_df.iterrows():
            ax.annotate(row['model_name'], 
                       xy=(row['tokens_per_correct'], row['latency_ms']),
                       xytext=(0, -8), textcoords='offset points',
                       fontsize=5, ha='center', va='top')
            ax.annotate(f"{int(row['avg_gen_tokens'])}", 
                       xy=(row['tokens_per_correct'], row['latency_ms']),
                       xytext=(0, 8), textcoords='offset points',
                       fontsize=5, ha='center', va='bottom')
        
        if pareto and len(dataset_df) > 1:
            x_vals = dataset_df['tokens_per_correct'].values
            y_vals = dataset_df['latency_ms'].values
            
            pareto_indices = calculate_pareto_frontier(x_vals, y_vals, minimize_x=True, minimize_y=True)
            
            if len(pareto_indices) > 1:
                pareto_x = x_vals[pareto_indices]
                pareto_y = y_vals[pareto_indices]
                
                sort_idx = np.argsort(pareto_x)
                ax.plot(pareto_x[sort_idx], pareto_y[sort_idx], 'r--', linewidth=2, alpha=0.8, label='Pareto Frontier')
        
        ax.set_xlabel('Tokens per Correct')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'{dataset}')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Model Size (B)', fontsize=8)
    
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Tokens per Correct vs Latency (Color: Model Size, Size: Tokens Generated)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_tokens_per_correct_vs_latency.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'pareto_tokens_per_correct_vs_latency.png'}")
    plt.close()
    
def plot_leaderboards(df: pd.DataFrame, output_dir: Path):
    """Plot 6 & 7: Leaderboard plots for accuracy and tokens_per_correct."""
    print("Creating leaderboard plots...")
    
    # Accuracy leaderboard
    plot_df = df.dropna(subset=['self_eval_acc', 'model_name', 'dataset']).copy()
    
    if len(plot_df) == 0:
        print("Warning: No data available for leaderboards")
        return

    datasets = sorted(plot_df['dataset'].unique())
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    
    if n_datasets == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        dataset_df = plot_df[plot_df['dataset'] == dataset].copy()
        
        if len(dataset_df) == 0:
            ax.set_visible(False)
            continue
        
        model_acc = dataset_df.groupby('model_name')['self_eval_acc'].mean().sort_values(ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_acc)))
        ax.barh(range(len(model_acc)), model_acc.values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(model_acc)))
        ax.set_yticklabels(model_acc.index, fontsize=8)
        ax.set_xlabel('Mean Accuracy', fontsize=9)
        ax.set_title(f'{dataset} - Accuracy', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for j, (model, value) in enumerate(model_acc.items()):
            ax.text(value + 0.01, j, f'{value:.3f}', va='center', fontsize=7)
    
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Model Accuracy Leaderboard', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'leaderboard_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'leaderboard_accuracy.png'}")
    plt.close()
    
    # Tokens per correct leaderboard
    plot_df_tpc = df.dropna(subset=['tokens_per_correct', 'model_name', 'dataset']).copy()
    
    if len(plot_df_tpc) == 0:
        print("Warning: No data available for tokens_per_correct leaderboards")
        return
    
    datasets_tpc = sorted(plot_df_tpc['dataset'].unique())
    n_datasets_tpc = len(datasets_tpc)
    n_cols_tpc = min(3, n_datasets_tpc)
    n_rows_tpc = (n_datasets_tpc + n_cols_tpc - 1) // n_cols_tpc
    
    fig, axes = plt.subplots(n_rows_tpc, n_cols_tpc, figsize=(6*n_cols_tpc, 4*n_rows_tpc))
    
    if n_datasets_tpc == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i, dataset in enumerate(datasets_tpc):
        ax = axes_flat[i]
        dataset_df = plot_df_tpc[plot_df_tpc['dataset'] == dataset].copy()
        
        if len(dataset_df) == 0:
            ax.set_visible(False)
            continue
        
        model_tpc = dataset_df.groupby('model_name')['tokens_per_correct'].mean().sort_values(ascending=False)
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(model_tpc)))
        ax.barh(range(len(model_tpc)), model_tpc.values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(model_tpc)))
        ax.set_yticklabels(model_tpc.index, fontsize=8)
        ax.set_xlabel('Mean Tokens per Correct (Lower is Better)', fontsize=9)
        ax.set_title(f'{dataset} - Efficiency', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for j, (model, value) in enumerate(model_tpc.items()):
            ax.text(value + (model_tpc.max() * 0.01), j, f'{value:.1f}', va='center', fontsize=7)
    
    for i in range(n_datasets_tpc, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Model Efficiency Leaderboard (Tokens per Correct)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'leaderboard_tokens_per_correct.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'leaderboard_tokens_per_correct.png'}")
    plt.close()
    
def export_datapoints_to_json(df: pd.DataFrame, output_dir: Path) -> None:
    """Export all datapoints used for plotting to JSON format."""
    print("Exporting datapoints to JSON...")
    
    plot_df = df.dropna(subset=['self_eval_acc', 'latency_ms', 'avg_gen_tokens'])
    
    if len(plot_df) == 0:
        print("Warning: No complete data available for JSON export")
        return
    
    datapoints = []
    for _, row in plot_df.iterrows():
        datapoint = {
            "model_name": row['model_name'],
            "model_family": row['model_family'],
            "dataset": row['dataset'],
            "runtime_ms": float(row['latency_ms']),
            "avg_gen_tokens": float(row['avg_gen_tokens']),
            "self_eval_accuracy": float(row['self_eval_acc'])
        }
        datapoints.append(datapoint)
    
    export_data = {
        "export_timestamp": pd.Timestamp.now().isoformat(),
        "total_datapoints": len(datapoints),
        "datapoints": datapoints
    }
    
    json_filepath = output_dir / "plot_datapoints.json"
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(datapoints)} datapoints to: {json_filepath}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate plots from wandb data')
    parser.add_argument('project_name', help='Wandb project name')
    parser.add_argument('--output-dir', default='plots', 
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--no-pareto', action='store_true',
                       help='Disable pareto frontier overlay')
    parser.add_argument('--no-export', action='store_true',
                       help='Disable JSON datapoints export')
    
    args = parser.parse_args()
    
    setup_plotting_style()
    
    output_dir = Path(args.output_dir) / args.project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df = query_wandb_data(args.project_name)
        
        print(f"\nData Summary:")
        print(f"Total runs: {len(df)}")
        print(f"Datasets: {sorted(df['dataset'].unique())}")
        print(f"Models: {sorted(df['model_name'].unique())}")
        
        print(f"\nData Availability:")
        print(f"Runs with latency_ms: {df['latency_ms'].notna().sum()}")
        print(f"Runs with self_eval_acc: {df['self_eval_acc'].notna().sum()}")
        print(f"Runs with avg_gen_tokens: {df['avg_gen_tokens'].notna().sum()}")
        print(f"Runs with params_B: {df['params_B'].notna().sum()}")
        print(f"Runs with release_date: {df['release_date'].notna().sum()}")
        print(f"Runs with tokens_per_correct: {df['tokens_per_correct'].notna().sum()}")
        print(f"Runs with energy_joules: {df['energy_joules'].notna().sum()}")
        
        if not args.no_export:
            export_datapoints_to_json(df, output_dir)
        
        print(f"\nGenerating plots...")
        
        # NEW PLOTS
        plot_accuracy_vs_energy(df, output_dir, pareto=not args.no_pareto)
        plot_accuracy_vs_release_date(df, output_dir)
        plot_accuracy_vs_tokens_per_correct(df, output_dir, pareto=not args.no_pareto)
        plot_efficiency_pareto(df, output_dir, pareto=not args.no_pareto)
        plot_leaderboards(df, output_dir)
        
        print(f"\n✅ All plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
