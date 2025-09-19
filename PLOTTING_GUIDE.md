# Plotting Guide

This guide explains how to generate high-quality plots from wandb data using the `generate_plots.py` script.

## Overview

The plotting script creates detailed scatter plots showing the relationship between model accuracy and performance metrics, with pareto frontier overlays to identify optimal trade-offs.

## Features

- **Accuracy vs Runtime**: Scatter plots showing accuracy vs average runtime per datapoint
- **Accuracy vs Token Generation**: Scatter plots showing accuracy vs average tokens generated
- **Pareto Frontier**: Overlay showing non-dominated solutions (minimize latency/tokens, maximize accuracy)
- **Log Scale X-Axis**: Better visualization of wide ranges in latency and token generation
- **Model Family Grouping**: Color-coded by model family, different shapes for individual models
- **Multi-Dataset Support**: Separate subplots for each dataset
- **High-Quality Output**: 300 DPI PNG files suitable for publications

## Usage

### Basic Usage

```bash
# Generate plots for a wandb project
python3 src/cli/generate_plots.py your-project-name
```

### Advanced Options

```bash
# Custom output directory
python3 src/cli/generate_plots.py your-project-name --output-dir custom_plots

# Disable pareto frontier overlay
python3 src/cli/generate_plots.py your-project-name --no-pareto
```

## Output

The script creates two main plot files in `plots/<project-name>/`:

1. **`accuracy_vs_latency_ms.png`**: Accuracy vs average runtime per datapoint
2. **`accuracy_vs_avg_gen_tokens.png`**: Accuracy vs average tokens generated

Each plot contains:
- **Subplots**: One for each dataset in your wandb project
- **Scatter Points**: Each point represents a model run (larger, more visible markers)
- **Colors**: Grouped by model family (e.g., llama, gemma, phi)
- **Shapes**: Different shapes for individual model names
- **Log Scale X-Axis**: Better visualization of wide ranges
- **Pareto Frontier**: Subtle red dashed line with descriptive label showing optimal trade-offs (minimize latency/tokens, maximize accuracy)
- **Model Annotations**: Clean model names (repository prefixes automatically removed) labeled on pareto frontier points with compact, non-intrusive text

## Required Wandb Data

The script expects the following parameters in your wandb runs:

### Config Parameters
- `model_family`: Model family (e.g., "llama", "gemma", "phi")
- `model`: Model name/identifier
- `dataset`: Dataset name

### Summary Metrics
- `latency_ms`: Average runtime per datapoint in milliseconds
- `self_eval_acc`: Self-evaluation accuracy score
- `avg_gen_tokens`: Average tokens generated per datapoint

## Data Filtering

The script automatically:
- Excludes runs with missing essential data
- Handles cases where parameters are missing
- Provides warnings for skipped runs
- Shows data availability summary

## Example Output Structure

```
plots/
└── your-project-name/
    ├── accuracy_vs_latency_ms.png
    └── accuracy_vs_avg_gen_tokens.png
```

## Troubleshooting

### Common Issues

1. **No valid runs found**: Check that your wandb project has runs with the required metrics
2. **Missing dependencies**: Ensure matplotlib, seaborn, and scipy are installed
3. **Authentication**: Make sure you're logged into wandb (`wandb login`)

### Data Requirements

- At least one run with complete data (latency_ms, self_eval_acc, avg_gen_tokens)
- Model family and name information in run config
- Dataset information in run config

## Customization

The script can be easily modified to:
- Change color schemes
- Add additional metrics
- Modify plot layouts
- Adjust pareto frontier calculation
- Add more plot types

See the source code in `src/cli/generate_plots.py` for implementation details.
