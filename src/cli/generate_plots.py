#!/usr/bin/env python3
"""
CLI script for generating high-quality plots from wandb data.

This script queries wandb runs from a specified project and creates detailed plots
showing accuracy vs runtime and accuracy vs token generation with pareto frontiers.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy.spatial import ConvexHull

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def setup_plotting_style():
    """Set up high-quality plotting style."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set high DPI for better quality
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # Set smaller font sizes to avoid covering the plot
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['legend.fontsize'] = 8

def query_wandb_data(project_name: str) -> pd.DataFrame:
    """
    Query wandb data for a given project and return as DataFrame.
    
    Args:
        project_name: Name of the wandb project
        
    Returns:
        DataFrame with run data
    """
    print(f"Querying wandb data for project: {project_name}")
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all runs from the project
    runs = api.runs(project_name)
    
    data = []
    for run in runs:
        try:
            # Extract config and summary data
            config = run.config
            summary = run.summary
            
            # Extract required parameters
            model_name = config.get('model_name', config.get('model', 'unknown'))
            # Clean model name by removing repository prefix (everything before first '/')
            if '/' in model_name:
                model_name = model_name.split('/', 1)[1]
            
            run_data = {
                'run_id': run.id,
                'run_name': run.name,
                'model_family': config.get('model_family', 'unknown'),
                'model_name': model_name,  # Clean model name without repo prefix
                'model_repo': config.get('model', 'unknown'),  # Keep repo name for reference
                'dataset': config.get('dataset', 'unknown'),
                'latency_ms': summary.get('latency_ms', None),
                'self_eval_acc': summary.get('self_eval_acc', None),
                'avg_gen_tokens': summary.get('avg_gen_tokens', None),
                'think_budget': config.get('think_budget', None),
                'style': config.get('style', None),
                'K': config.get('K', None),
                'batch_size': config.get('batch_size', None),
            }
            
            # Only add runs that have the essential data
            if (run_data['latency_ms'] is not None and 
                run_data['self_eval_acc'] is not None and 
                run_data['avg_gen_tokens'] is not None):
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
    """
    Calculate pareto frontier points.
    
    For our use case: minimize x (latency/tokens) AND maximize y (accuracy)
    
    Args:
        x: X-axis values (latency_ms or avg_gen_tokens)
        y: Y-axis values (accuracy)
        minimize_x: Whether to minimize x (True for latency/tokens)
        minimize_y: Whether to minimize y (False for accuracy - we want to maximize)
        
    Returns:
        Array of indices of pareto frontier points
    """
    if len(x) == 0:
        return np.array([])
    
    # For pareto frontier: we want to minimize x and maximize y
    # So we look for points where no other point has both lower x AND higher y
    
    pareto_indices = []
    
    for i in range(len(x)):
        is_pareto = True
        for j in range(len(x)):
            if i != j:
                # Check if point j dominates point i
                # j dominates i if: x_j <= x_i AND y_j >= y_i (with at least one strict inequality)
                if (x[j] <= x[i] and y[j] >= y[i]) and (x[j] < x[i] or y[j] > y[i]):
                    is_pareto = False
                    break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return np.array(pareto_indices)

def create_color_mapping(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Create color and shape mappings for model families and names.
    
    Args:
        df: DataFrame with model data
        
    Returns:
        Tuple of (family_colors, name_shapes) dictionaries
    """
    # Get unique families and names
    families = sorted(df['model_family'].unique())
    names = sorted(df['model_name'].unique())
    
    # Create color palette for families
    family_colors = {}
    base_colors = sns.color_palette("husl", len(families))
    
    for i, family in enumerate(families):
        family_colors[family] = base_colors[i]
    
    # Create shape mapping for model names
    shapes = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '+', 'x', 'X']
    name_shapes = {}
    
    for i, name in enumerate(names):
        name_shapes[name] = shapes[i % len(shapes)]
    
    return family_colors, name_shapes

def plot_accuracy_vs_metric(df: pd.DataFrame, metric: str, metric_label: str, 
                           output_dir: Path, pareto: bool = True):
    """
    Create accuracy vs metric plots for each dataset.
    
    Args:
        df: DataFrame with run data
        metric: Column name for the metric (e.g., 'latency_ms', 'avg_gen_tokens')
        metric_label: Label for the metric axis
        output_dir: Directory to save plots
        pareto: Whether to include pareto frontier
    """
    # Filter out runs with missing data
    plot_df = df.dropna(subset=['self_eval_acc', metric])
    
    if len(plot_df) == 0:
        print(f"Warning: No data available for {metric_label} plots")
        return
    
    # Get unique datasets
    datasets = sorted(plot_df['dataset'].unique())
    
    # Create color and shape mappings
    family_colors, name_shapes = create_color_mapping(plot_df)
    
    # Create subplots
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle different subplot configurations
    if n_datasets == 1:
        axes_flat = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes_flat = axes
    elif n_rows > 1 and n_cols == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()
    
    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        dataset_df = plot_df[plot_df['dataset'] == dataset]
        
        # Plot each model
        for _, row in dataset_df.iterrows():
            family = row['model_family']
            name = row['model_name']
            x_val = row[metric]
            y_val = row['self_eval_acc']
            
            color = family_colors[family]
            shape = name_shapes[name]
            
            ax.scatter(x_val, y_val, c=[color], marker=shape, s=60, 
                      alpha=0.7, edgecolors='black', linewidth=0.5,
                      label=f"{family} - {name}")
        
        # Add pareto frontier if requested
        if pareto and len(dataset_df) > 1:
            x_vals = dataset_df[metric].values
            y_vals = dataset_df['self_eval_acc'].values
            
            # Calculate pareto frontier (minimize latency/tokens, maximize accuracy)
            pareto_indices = calculate_pareto_frontier(x_vals, y_vals, 
                                                     minimize_x=True, minimize_y=False)
            
            if len(pareto_indices) > 1:
                pareto_x = x_vals[pareto_indices]
                pareto_y = y_vals[pareto_indices]
                
                # Sort for plotting
                sort_indices = np.argsort(pareto_x)
                pareto_x = pareto_x[sort_indices]
                pareto_y = pareto_y[sort_indices]
                pareto_indices_sorted = pareto_indices[sort_indices]
                
                ax.plot(pareto_x, pareto_y, 'r--', linewidth=1.5, alpha=0.6, 
                       label='Pareto Frontier (min latency/tokens, max accuracy)')
                
                # Add model name annotations for pareto frontier points
                for i, (x_val, y_val, idx) in enumerate(zip(pareto_x, pareto_y, pareto_indices_sorted)):
                    # Use the actual model name from the run, not the repo name
                    model_name = dataset_df.iloc[idx]['model_name']
                    # Position annotation slightly offset from the point
                    ax.annotate(model_name, (x_val, y_val), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=5, alpha=0.8,
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Customize plot
        ax.set_xlabel(metric_label)
        ax.set_ylabel('Accuracy (Self-Eval)')
        ax.set_title(f'{dataset}')
        ax.grid(True, alpha=0.3)
        
        # Set log scale for x-axis
        ax.set_xscale('log')

        #ax.set_yscale('log')
        
        # Set reasonable axis limits
        if len(dataset_df) > 0:
            x_min = dataset_df[metric].min()
            x_max = dataset_df[metric].max()
            y_margin = (dataset_df['self_eval_acc'].max() - dataset_df['self_eval_acc'].min()) * 0.1
            
            # For log scale, use multiplicative margins
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(dataset_df['self_eval_acc'].min() - y_margin, 
                       dataset_df['self_eval_acc'].max() + y_margin)
    
    # Hide unused subplots
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Create legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        # Remove duplicate labels
        unique_labels = []
        unique_handles = []
        seen = set()
        for handle, label in zip(handles, labels):
            if label not in seen:
                unique_handles.append(handle)
                unique_labels.append(label)
                seen.add(label)
        
        fig.legend(unique_handles, unique_labels, 
                  bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"accuracy_vs_{metric}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    
    plt.close()

def export_datapoints_to_json(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export all datapoints used for plotting to JSON format.
    
    Args:
        df: DataFrame with run data
        output_dir: Directory to save JSON file
    """
    print("Exporting datapoints to JSON...")
    
    # Filter out runs with missing essential data (same as plotting)
    plot_df = df.dropna(subset=['self_eval_acc', 'latency_ms', 'avg_gen_tokens'])
    
    if len(plot_df) == 0:
        print("Warning: No complete data available for JSON export")
        return
    
    # Convert DataFrame to list of dictionaries with only the required fields
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
    
    # Create export structure
    export_data = {
        "export_timestamp": pd.Timestamp.now().isoformat(),
        "total_datapoints": len(datapoints),
        "datapoints": datapoints
    }
    
    # Save to JSON file
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
    
    # Set up plotting style
    setup_plotting_style()
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Query wandb data
        df = query_wandb_data(args.project_name)
        
        # Print data summary
        print(f"\nData Summary:")
        print(f"Total runs: {len(df)}")
        print(f"Datasets: {sorted(df['dataset'].unique())}")
        print(f"Model families: {sorted(df['model_family'].unique())}")
        print(f"Models: {sorted(df['model_name'].unique())}")
        
        # Check data availability
        print(f"\nData Availability:")
        print(f"Runs with latency_ms: {df['latency_ms'].notna().sum()}")
        print(f"Runs with self_eval_acc: {df['self_eval_acc'].notna().sum()}")
        print(f"Runs with avg_gen_tokens: {df['avg_gen_tokens'].notna().sum()}")
        
        # Export datapoints to JSON (unless disabled)
        if not args.no_export:
            export_datapoints_to_json(df, output_dir)
        
        # Create plots
        print(f"\nGenerating plots...")
        
        # Accuracy vs Latency plots
        plot_accuracy_vs_metric(df, 'latency_ms', 'Average Runtime (ms)', 
                               output_dir, pareto=not args.no_pareto)
        
        # Accuracy vs Tokens plots  
        plot_accuracy_vs_metric(df, 'avg_gen_tokens', 'Average Generated Tokens',
                               output_dir, pareto=not args.no_pareto)
        
        print(f"\n✅ All plots and datapoints saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
