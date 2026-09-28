"""Best-effort description of the benchmark machine (docs/benchmark-protocol.md, §4).

Nothing here changes the machine. Fields the OS does not provide are ``None``,
and a lookup failure is recorded as a string instead of stopping the run.
"""

from __future__ import annotations

import importlib
import os
import platform
import struct
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["cpu_info", "environment_record", "parse_cpuinfo"]

_RELATION_PROCESSOR_CORE = 0


def _windows_cpu() -> dict[str, Any]:
    if sys.platform != "win32":
        return {}
    import ctypes
    import winreg

    info: dict[str, Any] = {}
    key = winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    )
    with key:
        info["model"] = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        info["nominal_mhz"] = int(winreg.QueryValueEx(key, "~MHz")[0])
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_uint32(0)
    kernel32.GetLogicalProcessorInformationEx(_RELATION_PROCESSOR_CORE, None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel32.GetLogicalProcessorInformationEx(
        _RELATION_PROCESSOR_CORE, buffer, ctypes.byref(size)
    ):
        raise OSError(ctypes.get_last_error(), "GetLogicalProcessorInformationEx failed")
    raw, offset, cores = buffer.raw[: size.value], 0, 0
    while offset + 8 <= len(raw):
        relationship, record_size = struct.unpack_from("<II", raw, offset)
        if record_size == 0:
            break
        cores += relationship == _RELATION_PROCESSOR_CORE
        offset += record_size
    info["physical_cores"] = cores or None
    info["physical_cores_source"] = "GetLogicalProcessorInformationEx(RelationProcessorCore)"
    return info


def parse_cpuinfo(text: str) -> dict[str, Any]:
    """Model, MHz, and physical core count from Linux ``/proc/cpuinfo`` text."""
    info: dict[str, Any] = {}
    cores: set[tuple[str, str]] = set()
    physical_id = core_id = None
    for line in [*text.splitlines(), ""]:
        if not line.strip():
            if physical_id is not None and core_id is not None:
                cores.add((physical_id, core_id))
            physical_id = core_id = None
            continue
        key, _, value = (part.strip() for part in line.partition(":"))
        if key == "model name":
            info.setdefault("model", value)
        elif key == "cpu MHz":
            info.setdefault("nominal_mhz", float(value))
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            core_id = value
    info["physical_cores"] = len(cores) or None
    info["physical_cores_source"] = "/proc/cpuinfo (physical id, core id)"
    return info


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "model": None,
        "nominal_mhz": None,
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    try:
        if sys.platform == "win32":
            info.update(_windows_cpu())
        elif sys.platform.startswith("linux"):
            info.update(
                parse_cpuinfo(Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace"))
            )
    except (OSError, ValueError) as exc:
        info["lookup_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _version(name: str) -> str | None:
    try:
        return str(importlib.import_module(name).__version__)
    except ImportError:
        return None


def environment_record() -> dict[str, Any]:
    """OS, CPU, interpreter, library versions, thread pools, and the timer."""
    import torch

    clock = time.get_clock_info("perf_counter")
    record: dict[str, Any] = {
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu": cpu_info(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {name: _version(name) for name in ("onnxruntime", "torch", "numpy", "onnx")},
        "torch_threads": {
            "intra_op_default": torch.get_num_threads(),
            "inter_op": torch.get_num_interop_threads(),
        },
        "timer": {
            "function": "time.perf_counter_ns",
            "implementation": clock.implementation,
            "resolution_s": clock.resolution,
            "monotonic": clock.monotonic,
        },
    }
    try:
        import onnxruntime

        record["onnxruntime_providers"] = list(onnxruntime.get_available_providers())
    except ImportError:
        record["onnxruntime_providers"] = None
    return record
