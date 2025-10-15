import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.metrics.energy import EnergyMeter, _NVML_AVAILABLE


class TestEnergyMeter:
    def test_energy_meter_disabled_when_nvml_unavailable(self):
        """Test EnergyMeter disables when NVML is not available"""
        with patch('src.metrics.energy._NVML_AVAILABLE', False):
            meter = EnergyMeter()
            assert not meter.enabled
            assert meter.total_joules == 0.0

    @patch('src.metrics.energy._NVML_AVAILABLE', True)
    def test_energy_meter_initialization_with_nvml(self):
        """Test EnergyMeter initialization when NVML is available"""
        mock_device = Mock()

        with patch('src.metrics.energy.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_device
            mock_pynvml.nvmlDeviceGetTotalEnergyConsumption.return_value = 1000  # mJ

            meter = EnergyMeter(device_index=1)
            assert meter.enabled
            assert meter.device_index == 1
            assert meter._use_energy_counter
            mock_pynvml.nvmlInit.assert_called_once()

    @patch('src.metrics.energy._NVML_AVAILABLE', True)
    def test_energy_meter_fallback_to_power_sampling(self):
        """Test EnergyMeter falls back to power sampling when energy counter unavailable"""
        mock_device = Mock()

        with patch('src.metrics.energy.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_device
            mock_pynvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = Exception("Not supported")

            meter = EnergyMeter()
            assert meter.enabled
            assert not meter._use_energy_counter

    @patch('src.metrics.energy._NVML_AVAILABLE', True)
    def test_energy_meter_nvml_init_failure(self):
        """Test EnergyMeter handles NVML initialization failure gracefully"""
        with patch('src.metrics.energy.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.side_effect = Exception("NVML init failed")

            meter = EnergyMeter()
            assert not meter.enabled
            assert meter._device is None

    def test_energy_meter_disabled_operations(self):
        """Test EnergyMeter operations when disabled"""
        meter = EnergyMeter(enabled=False)

        # These should be no-ops
        meter.start()
        result = meter.stop()
        meter.close()

        assert result == 0.0
        assert meter.total_joules == 0.0

    @patch('src.metrics.energy._NVML_AVAILABLE', True)
    def test_energy_meter_energy_counter_mode(self):
        """Test EnergyMeter using energy counter mode"""
        mock_device = Mock()

        energy_values = [500, 1000, 2500, 4000]  # mJ values: init check, start, stop, next start
        energy_iter = iter(energy_values)

        with patch('src.metrics.energy.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_device
            # Set up side_effect before EnergyMeter creation so availability check succeeds
            mock_pynvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = lambda *args: next(energy_iter)

            meter = EnergyMeter()

            meter.start()
            energy_delta = meter.stop()

            assert energy_delta == 1.5  # 2.5 - 1.0
            assert meter.total_joules == 1.5

    @patch('src.metrics.energy._NVML_AVAILABLE', True)
    def test_energy_meter_power_sampling_mode(self):
        """Test EnergyMeter using power sampling mode"""
        mock_device = Mock()

        with patch('src.metrics.energy.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_device
            mock_pynvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = Exception("Not supported")
            mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = [1000, 1000, 2000, 2000]  # mW

            meter = EnergyMeter(sample_interval_s=0.1)

            with patch('time.perf_counter', side_effect=[0.0, 0.1, 0.2, 0.3]):
                with patch('time.sleep') as mock_sleep:
                    meter.start()
                    time.sleep(0.1)  # This will be mocked
                    energy_delta = meter.stop()

            # Power samples: 1.0W, 1.0W, 2.0W, 2.0W over 0.1s intervals
            # Energy = (1.0+1.0)*0.1/2 + (1.0+2.0)*0.1/2 + (2.0+2.0)*0.1/2 = 0.1 + 0.15 + 0.2 = 0.45J
            expected_energy = 0.45
            assert energy_delta == pytest.approx(expected_energy, abs=1e-6)

    def test_energy_meter_context_manager(self):
        """Test EnergyMeter context manager usage"""
        meter = EnergyMeter(enabled=False)

        with meter.measure() as ctx:
            assert ctx is not None
            # In disabled mode, this should be a no-op

        assert meter.total_joules == 0.0

    @patch('src.metrics.energy._NVML_AVAILABLE', True)
    def test_energy_meter_close_handles_exceptions(self):
        """Test EnergyMeter close method handles exceptions gracefully"""
        mock_device = Mock()

        with patch('src.metrics.energy.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_device
            mock_pynvml.nvmlDeviceGetTotalEnergyConsumption.return_value = 1000
            mock_pynvml.nvmlShutdown.side_effect = Exception("Shutdown failed")

            meter = EnergyMeter()
            # Should not raise exception
            meter.close()

    def test_energy_meter_custom_sample_interval(self):
        """Test EnergyMeter with custom sample interval"""
        meter = EnergyMeter(sample_interval_s=0.05, enabled=False)
        assert meter.sample_interval_s == 0.05

    def test_energy_meter_device_index(self):
        """Test EnergyMeter with custom device index"""
        meter = EnergyMeter(device_index=2, enabled=False)
        assert meter.device_index == 2

    @patch('src.metrics.energy._NVML_AVAILABLE', True)
    def test_energy_meter_power_sampling_integration_bounds(self):
        """Test EnergyMeter power sampling integration handles edge cases"""
        mock_device = Mock()

        with patch('src.metrics.energy.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_device
            mock_pynvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = Exception("Not supported")
            mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 1000

            meter = EnergyMeter()

            # Manually test integration with insufficient samples
            meter._samples = [(0.0, 1.0)]  # Only one sample
            result = meter._integrate_samples_j()
            assert result == 0.0  # Should return 0 with < 2 samples

            # Test with two samples
            meter._samples = [(0.0, 1.0), (1.0, 2.0)]
            result = meter._integrate_samples_j()
            expected = 0.5 * (1.0 + 2.0) * 1.0  # Trapezoidal integration
            assert result == pytest.approx(expected, abs=1e-6)

    @patch('src.metrics.energy._NVML_AVAILABLE', True)
    def test_energy_meter_power_sampling_exception_handling(self):
        """Test EnergyMeter handles exceptions during power sampling"""
        mock_device = Mock()

        with patch('src.metrics.energy.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_device
            mock_pynvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = Exception("Not supported")
            mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception("Power read failed")

            meter = EnergyMeter()

            meter.start()
            energy_delta = meter.stop()

            # Should handle exception gracefully and return 0
            assert energy_delta == 0.0
