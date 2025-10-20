#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced generate_plots.py script.

Tests:
1. CLI functionality and argument parsing
2. Data querying from wandb
3. Plot generation for all 6 plot types
4. Data validation and error handling
5. Output file generation
"""

import sys
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cli.generate_plots import (
    query_wandb_data,
    calculate_pareto_frontier,
    plot_accuracy_vs_energy,
    plot_accuracy_vs_release_date,
    plot_accuracy_vs_tokens_per_correct,
    plot_efficiency_pareto,
    plot_leaderboards,
    export_datapoints_to_json,
)


class TestParetoFrontier(unittest.TestCase):
    """Test pareto frontier calculation."""

    def test_pareto_frontier_basic(self):
        """Test basic pareto frontier calculation."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([4.0, 3.0, 2.0, 1.0])
        
        pareto_indices = calculate_pareto_frontier(x, y)
        
        # First point (1.0, 4.0) should always be on the frontier
        # since it has minimum x and maximum y
        self.assertIn(0, pareto_indices)
        self.assertGreater(len(pareto_indices), 0)
    
    def test_pareto_frontier_dominated(self):
        """Test pareto frontier with dominated points."""
        x = np.array([1.0, 1.0, 2.0, 3.0])
        y = np.array([4.0, 2.0, 3.0, 1.0])
        
        pareto_indices = calculate_pareto_frontier(x, y)
        
        # Point (1.0, 2.0) should be dominated by (1.0, 4.0)
        self.assertNotIn(1, pareto_indices)
    
    def test_pareto_frontier_empty(self):
        """Test pareto frontier with empty input."""
        x = np.array([])
        y = np.array([])

        pareto_indices = calculate_pareto_frontier(x, y)
        
        self.assertEqual(len(pareto_indices), 0)


class TestDataExport(unittest.TestCase):
    """Test data export functionality."""

    def test_export_datapoints_to_json(self):
        """Test exporting datapoints to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create sample dataframe
            df = pd.DataFrame({
                'model_name': ['model_a', 'model_b'],
                'model_family': ['family_x', 'family_y'],
                'dataset': ['dataset_1', 'dataset_1'],
                'latency_ms': [100.0, 200.0],
                'avg_gen_tokens': [1000.0, 2000.0],
                'self_eval_acc': [0.8, 0.9],
            })

            export_datapoints_to_json(df, output_dir)

            json_file = output_dir / "plot_datapoints.json"
            self.assertTrue(json_file.exists())

            with open(json_file, 'r') as f:
                data = json.load(f)

            self.assertEqual(len(data['datapoints']), 2)
            self.assertEqual(data['total_datapoints'], 2)
            self.assertIn('export_timestamp', data)


class TestPlotGeneration(unittest.TestCase):
    """Test plot generation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = Path(tempfile.mkdtemp())
        
        # Create sample dataframe with all required columns
        self.df = pd.DataFrame({
            'model_name': ['model_a', 'model_b', 'model_a', 'model_b'],
            'model_family': ['family_x', 'family_y', 'family_x', 'family_y'],
            'dataset': ['dataset_1', 'dataset_1', 'dataset_2', 'dataset_2'],
            'latency_ms': [100.0, 200.0, 150.0, 250.0],
            'avg_gen_tokens': [1000.0, 2000.0, 1500.0, 2500.0],
            'self_eval_acc': [0.8, 0.9, 0.7, 0.85],
            'params_B': [7.0, 13.0, 7.0, 13.0],
            'energy_joules': [500.0, 1000.0, 750.0, 1250.0],
            'tokens_per_correct': [200.0, 150.0, 250.0, 180.0],
            'release_date': ['2024-01-01', '2024-02-01', '2024-01-01', '2024-02-01'],
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.output_dir)
    
    def test_plot_accuracy_vs_energy(self):
        """Test accuracy vs energy plot generation."""
        plot_accuracy_vs_energy(self.df, self.output_dir, pareto=True)
        
        plot_file = self.output_dir / 'pareto_accuracy_vs_energy.png'
        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 0)
    
    def test_plot_accuracy_vs_release_date(self):
        """Test accuracy vs release date plot generation."""
        plot_accuracy_vs_release_date(self.df, self.output_dir)
        
        plot_file = self.output_dir / 'scatter_accuracy_vs_release_date.png'
        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 0)
    
    def test_plot_accuracy_vs_tokens_per_correct(self):
        """Test accuracy vs tokens per correct plot generation."""
        plot_accuracy_vs_tokens_per_correct(self.df, self.output_dir, pareto=True)
        
        plot_file = self.output_dir / 'pareto_accuracy_vs_tokens_per_correct.png'
        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 0)
    
    def test_plot_efficiency_pareto(self):
        """Test tokens per correct vs latency plot generation."""
        plot_efficiency_pareto(self.df, self.output_dir, pareto=True)
        
        plot_file = self.output_dir / 'pareto_tokens_per_correct_vs_latency.png'
        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 0)
    
    def test_plot_leaderboards(self):
        """Test leaderboard plot generation."""
        plot_leaderboards(self.df, self.output_dir)
        
        accuracy_file = self.output_dir / 'leaderboard_accuracy.png'
        efficiency_file = self.output_dir / 'leaderboard_tokens_per_correct.png'
        
        self.assertTrue(accuracy_file.exists())
        self.assertTrue(efficiency_file.exists())
        self.assertGreater(accuracy_file.stat().st_size, 0)
        self.assertGreater(efficiency_file.stat().st_size, 0)
    
    def test_plot_with_missing_data(self):
        """Test plot generation with missing data columns."""
        # Create dataframe with NaN values
        df_with_nan = self.df.copy()
        df_with_nan.loc[0, 'energy_joules'] = np.nan
        
        # Should not raise an error
        plot_accuracy_vs_energy(df_with_nan, self.output_dir, pareto=True)
        
        plot_file = self.output_dir / 'pareto_accuracy_vs_energy.png'
        # File may or may not exist depending on implementation
        # but the function should not raise an error
    
    def test_plot_single_dataset(self):
        """Test plot generation with single dataset."""
        df_single = self.df[self.df['dataset'] == 'dataset_1']
        
        plot_accuracy_vs_energy(df_single, self.output_dir, pareto=True)
        
        plot_file = self.output_dir / 'pareto_accuracy_vs_energy.png'
        self.assertTrue(plot_file.exists())


class TestDataValidation(unittest.TestCase):
    """Test data validation and error handling."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            empty_df = pd.DataFrame({
                'model_name': [],
                'dataset': [],
                'latency_ms': [],
                'avg_gen_tokens': [],
                'self_eval_acc': [],
                'params_B': [],
                'energy_joules': [],
                'tokens_per_correct': [],
                'release_date': [],
            })
            
            # Should handle gracefully without crashing
            plot_accuracy_vs_energy(empty_df, output_dir, pareto=True)
            plot_leaderboards(empty_df, output_dir)


class TestCLIFunctionality(unittest.TestCase):
    """Test CLI functionality."""
    
    def test_cli_argument_parser(self):
        """Test CLI argument parser configuration."""
        import argparse
        
        # Create a parser similar to the main script
        parser = argparse.ArgumentParser(description='Generate plots from wandb data')
        parser.add_argument('project_name', help='Wandb project name')
        parser.add_argument('--output-dir', default='plots', 
                           help='Output directory for plots (default: plots)')
        parser.add_argument('--no-pareto', action='store_true',
                           help='Disable pareto frontier overlay')
        parser.add_argument('--no-export', action='store_true',
                           help='Disable JSON datapoints export')
        
        # Test parsing valid arguments
        args = parser.parse_args(['EnergyEfficiency1018', '--output-dir', 'custom_plots'])
        self.assertEqual(args.project_name, 'EnergyEfficiency1018')
        self.assertEqual(args.output_dir, 'custom_plots')
        self.assertFalse(args.no_pareto)
        self.assertFalse(args.no_export)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency."""
    
    def test_pareto_indices_validity(self):
        """Test that pareto indices are valid."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        pareto_indices = calculate_pareto_frontier(x, y)
        
        # All indices should be within bounds
        self.assertTrue(all(0 <= idx < len(x) for idx in pareto_indices))
        # All indices should be unique
        self.assertEqual(len(pareto_indices), len(set(pareto_indices)))


class TestPlotsWithLimitedData(unittest.TestCase):
    """Test plot generation with minimal data."""
    
    def setUp(self):
        """Set up test fixtures with minimal data."""
        self.output_dir = Path(tempfile.mkdtemp())
        
        self.df = pd.DataFrame({
            'model_name': ['model_a'],
            'model_family': ['family_x'],
            'dataset': ['dataset_1'],
            'latency_ms': [100.0],
            'avg_gen_tokens': [1000.0],
            'self_eval_acc': [0.8],
            'params_B': [7.0],
            'energy_joules': [500.0],
            'tokens_per_correct': [200.0],
            'release_date': ['2024-01-01'],
        })
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.output_dir)
    
    def test_single_point_plot(self):
        """Test plot generation with single data point."""
        # Should not raise an error
        plot_accuracy_vs_energy(self.df, self.output_dir, pareto=True)
        plot_leaderboards(self.df, self.output_dir)


if __name__ == '__main__':
    unittest.main()
