#!/bin/bash
# Example script for generating plots from wandb data

# Activate virtual environment
source .venv/bin/activate

# Generate plots for a specific wandb project
# Replace 'your-project-name' with your actual wandb project name
python3 src/cli/generate_plots.py your-project-name

# Generate plots with custom output directory
python3 src/cli/generate_plots.py your-project-name --output-dir custom_plots

# Generate plots without pareto frontier
python3 src/cli/generate_plots.py your-project-name --no-pareto

echo "Plots will be saved to: plots/your-project-name/"
echo "  - accuracy_vs_latency_ms.png"
echo "  - accuracy_vs_avg_gen_tokens.png"
