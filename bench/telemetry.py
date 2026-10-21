"""
Hardware telemetry capture for benchmarking:
Captures GPU and RAM metrics before and after each query using pynvml and psutil.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def get_telemetry_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "gpu_temp": None,
        "gpu_clock": None,
        "gpu_mem_used_mb": None,
        "gpu_util_pct": None,
        "ram_used_pct": None,
        "ram_used_mb": None,
    }

    # RAM telemetry via psutil
    try:
        import psutil
        vmem = psutil.virtual_memory()
        snapshot["ram_used_pct"] = round(vmem.percent, 1)
        snapshot["ram_used_mb"] = round(vmem.used / (1024 ** 2), 1)
    except Exception:
        pass

    # GPU telemetry via pynvml
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            snapshot["gpu_temp"] = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            snapshot["gpu_clock"] = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            snapshot["gpu_mem_used_mb"] = round(mem.used / (1024 ** 2), 1)
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                snapshot["gpu_util_pct"] = int(rates.gpu)
            except Exception:
                pass
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return snapshot
