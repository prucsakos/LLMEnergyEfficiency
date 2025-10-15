from __future__ import annotations
import time, threading
from typing import Optional
try:
    import pynvml  # NVML
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

class EnergyMeter:
    """GPU energy meter using NVIDIA NVML.
    Prefers the total energy counter; falls back to power sampling.
    Usage:
        meter = EnergyMeter()              # auto-enables if NVML available
        with meter.measure():
            ... generation work ...
        print(meter.total_joules)
    """
    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.02, enabled: Optional[bool] = None):
        self.device_index = device_index
        self.sample_interval_s = float(sample_interval_s)
        self.enabled = (enabled if enabled is not None else _NVML_AVAILABLE)
        self._device = None
        self._use_energy_counter = False
        self._sampler_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._samples = []  # (t, power_w)
        self._start_energy_j = None
        self._end_energy_j = None
        self.total_joules: float = 0.0

        # Get logger for initialization status logging
        try:
            from ..logs.benchmark_logger import get_logger
            logger = get_logger()
        except ImportError:
            # If logger import fails, don't log anything
            logger = None

        # Log initialization status
        if logger:
            if not self.enabled:
                logger.info(f"⚡ EnergyMeter: DISABLED (NVML not available or explicitly disabled)")
            elif not _NVML_AVAILABLE:
                logger.info(f"⚡ EnergyMeter: DISABLED (pynvml not installed)")
            else:
                logger.info(f"⚡ EnergyMeter: INITIALIZING on device {device_index}")

        if self.enabled and _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._device = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                try:
                    _ = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._device)
                    self._use_energy_counter = True
                    if logger:
                        logger.info(f"⚡ EnergyMeter: ENABLED with energy counters (joules measurement)")
                except Exception:
                    self._use_energy_counter = False
                    if logger:
                        logger.info(f"⚡ EnergyMeter: ENABLED with power sampling (watt estimation, {sample_interval_s*1000:.0f}ms intervals)")
            except Exception as e:
                self.enabled = False
                self._device = None
                if logger:
                    logger.warning(f"⚡ EnergyMeter: FAILED to initialize NVML - {e}")
        elif logger and self.enabled and not _NVML_AVAILABLE:
            logger.info(f"⚡ EnergyMeter: DISABLED (pynvml not available)")

    def _read_energy_counter_j(self) -> float:
        # NVML returns millijoules on most GPUs.
        mJ = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._device)
        return float(mJ) / 1000.0

    def _read_power_w(self) -> float:
        # NVML returns milliwatts for getPowerUsage
        mW = pynvml.nvmlDeviceGetPowerUsage(self._device)
        return float(mW) / 1000.0

    def start(self):
        if not self.enabled:
            return
        if self._use_energy_counter:
            self._start_energy_j = self._read_energy_counter_j()
        else:
            self._samples.clear()
            self._stop.clear()
            self._sampler_thread = threading.Thread(target=self._sampler_loop, daemon=True)
            self._sampler_thread.start()

    def stop(self) -> float:
        if not self.enabled:
            return 0.0
        if self._use_energy_counter:
            self._end_energy_j = self._read_energy_counter_j()
            delta_j = max(0.0, float(self._end_energy_j - (self._start_energy_j or 0.0)))
        else:
            self._stop.set()
            if self._sampler_thread and self._sampler_thread.is_alive():
                self._sampler_thread.join(timeout=1.0)
            delta_j = self._integrate_samples_j()
        self.total_joules += delta_j
        return float(delta_j)

    def _sampler_loop(self):
        while not self._stop.is_set():
            try:
                p_w = self._read_power_w()
            except Exception:
                break
            t = time.perf_counter()
            self._samples.append((t, p_w))
            time.sleep(self.sample_interval_s)
        # Final sample for clean integration
        try:
            p_w = self._read_power_w()
            t = time.perf_counter()
            self._samples.append((t, p_w))
        except Exception:
            pass

    def _integrate_samples_j(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        energy_j = 0.0
        for (t0, p0), (t1, p1) in zip(self._samples[:-1], self._samples[1:]):
            dt = max(0.0, t1 - t0)
            energy_j += 0.5 * (p0 + p1) * dt
        return float(energy_j)

    def measure(self):
        class _Ctx:
            def __init__(self, outer: 'EnergyMeter'): self.outer = outer
            def __enter__(self): self.outer.start(); return self
            def __exit__(self, *exc): self.outer.stop(); return False
        return _Ctx(self)

    def close(self):
        if self.enabled and _NVML_AVAILABLE:
            try: pynvml.nvmlShutdown()
            except Exception: pass
