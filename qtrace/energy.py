"""Measure GPU energy for a block of code.

NVIDIA GPUs keep a running counter of the total energy they have used, in
millijoules. We read it before and after the block and subtract. This is the
same counter used by the Zeus and TokenPowerBench measurement tools.

Usage:
    with EnergyMeter() as meter:
        ... run inference ...
    print(meter.joules, meter.seconds)

Limits you must state in the paper:
  * GPU only. CPU and RAM energy are not included (EC2 instances do not expose
    the CPU's RAPL counters to guests).
  * The number includes the GPU's idle draw during the block. Call
    measure_idle_watts() once per machine and report both raw and idle-subtracted.
  * On machines without an NVIDIA GPU, joules is None.
"""

import time

import torch


class EnergyMeter:
    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.joules = None
        self.seconds = None
        self._handle = None

    def _read_millijoules(self):
        import pynvml

        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)

    def __enter__(self):
        if torch.cuda.is_available():
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            torch.cuda.synchronize()  # wait for queued GPU work so it is not billed to this block
            self._start_mj = self._read_millijoules()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        if self._handle is not None:
            torch.cuda.synchronize()  # wait until the block's GPU work has really finished
            self.joules = (self._read_millijoules() - self._start_mj) / 1000.0
        self.seconds = time.perf_counter() - self._start_time
        return False  # do not swallow exceptions


def measure_idle_watts(seconds=10.0, gpu_index=0):
    """Average GPU power while nothing is running. Returns None without an NVIDIA GPU."""
    if not torch.cuda.is_available():
        return None
    with EnergyMeter(gpu_index) as meter:
        time.sleep(seconds)
    return meter.joules / meter.seconds
