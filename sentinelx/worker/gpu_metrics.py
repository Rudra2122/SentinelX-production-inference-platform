import os
import time
import threading
from typing import Optional

from sentinelx.observability.metrics import (
    GPU_UTILIZATION_PERCENT,
    GPU_MEM_USED_BYTES,
    GPU_TEMP_CELSIUS,
    GPU_PRESENT,
    GPU_NVML_OK,
)

DEFAULT_POLL_SEC = float(os.getenv("SENTINELX_GPU_POLL_SEC", "2.0"))

def _safe_import_pynvml():
    try:
        import pynvml  # type: ignore
        return pynvml
    except Exception:
        return None

def _poll_loop(stop_event: threading.Event, poll_sec: float):
    pynvml = _safe_import_pynvml()
    if pynvml is None:
        # NVML library not available (or dependency missing)
        GPU_NVML_OK.set(0)
        GPU_PRESENT.set(0)
        return

    try:
        pynvml.nvmlInit()
        GPU_NVML_OK.set(1)
    except Exception:
        GPU_NVML_OK.set(0)
        GPU_PRESENT.set(0)
        return

    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception:
        GPU_PRESENT.set(0)
        return

    if device_count <= 0:
        GPU_PRESENT.set(0)
        return

    GPU_PRESENT.set(1)

    # Poll until stopped
    while not stop_event.is_set():
        for i in range(device_count):
            gpu_label = str(i)
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # has .gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)         # has .used
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                GPU_UTILIZATION_PERCENT.labels(gpu=gpu_label).set(float(util.gpu))
                GPU_MEM_USED_BYTES.labels(gpu=gpu_label).set(float(mem.used))
                GPU_TEMP_CELSIUS.labels(gpu=gpu_label).set(float(temp))

            except Exception:
                # Don’t crash the worker because NVML hiccuped
                continue

        stop_event.wait(poll_sec)

def start_gpu_metrics_poller(poll_sec: Optional[float] = None) -> threading.Event:
    """
    Starts a daemon thread that updates Prometheus gauges for GPU metrics.
    Returns a stop_event you can set() on shutdown if you want.
    """
    sec = poll_sec if poll_sec is not None else DEFAULT_POLL_SEC
    stop_event = threading.Event()
    t = threading.Thread(target=_poll_loop, args=(stop_event, sec), daemon=True)
    t.start()
    return stop_event
