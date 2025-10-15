import pytest
import numpy as np
from src.metrics.inequality import gini


class TestGini:
    def test_gini_perfect_equality(self):
        """Test Gini coefficient for perfect equality (all values equal)"""
        values = [1.0, 1.0, 1.0, 1.0]
        result = gini(values)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_gini_perfect_inequality(self):
        """Test Gini coefficient for perfect inequality (one value takes everything)"""
        values = [0.0, 0.0, 0.0, 1.0]
        result = gini(values)
        assert result == pytest.approx(0.75, abs=1e-10)  # Expected value for this distribution

    def test_gini_empty_list(self):
        """Test Gini coefficient for empty list"""
        values = []
        result = gini(values)
        assert result == 0.0

    def test_gini_single_value(self):
        """Test Gini coefficient for single value"""
        values = [5.0]
        result = gini(values)
        assert result == 0.0

    def test_gini_negative_values_raises_error(self):
        """Test that Gini coefficient raises error for negative values"""
        values = [1.0, -1.0, 2.0]
        with pytest.raises(ValueError, match="Gini only defined for non‑negative values"):
            gini(values)

    def test_gini_zero_values(self):
        """Test Gini coefficient when all values are zero"""
        values = [0.0, 0.0, 0.0]
        result = gini(values)
        assert result == 0.0

    def test_gini_typical_distribution(self):
        """Test Gini coefficient for a typical distribution"""
        values = [1, 2, 3, 4, 5]
        result = gini(values)
        # Expected value calculated manually or from known source
        expected = 0.26666666666666666  # Approximate expected value
        assert result == pytest.approx(expected, abs=1e-6)

    def test_gini_large_numbers(self):
        """Test Gini coefficient with large numbers"""
        values = [1000000, 2000000, 3000000]
        result = gini(values)
        # Should be same as normalized version
        normalized_values = [1, 2, 3]
        expected = gini(normalized_values)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_gini_iterator_input(self):
        """Test Gini coefficient accepts iterator input"""
        values_iter = iter([1.0, 2.0, 3.0, 4.0])
        result = gini(values_iter)
        values_list = [1.0, 2.0, 3.0, 4.0]
        expected = gini(values_list)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_gini_bounds(self):
        """Test that Gini coefficient is always between 0 and 1"""
        # Test various distributions
        test_cases = [
            [1, 1, 1],  # Perfect equality
            [1, 0, 0],  # High inequality
            [1, 2, 3, 4, 5],  # Moderate inequality
            [0, 0, 0, 100],  # Extreme inequality
        ]

        for values in test_cases:
            result = gini(values)
            assert 0.0 <= result <= 1.0, f"Gini {result} out of bounds [0,1] for values {values}"
