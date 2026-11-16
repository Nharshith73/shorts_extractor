"""Sample peak resident memory for this process and its FFmpeg descendants."""

import os
from pathlib import Path
import threading


def _windows_rss() -> int:
    import ctypes as c
    from ctypes import wintypes as w
    class Entry(c.Structure):
        _fields_ = [("dwSize", w.DWORD), ("cntUsage", w.DWORD), ("pid", w.DWORD), ("heap", c.c_size_t),
                    ("module", w.DWORD), ("threads", w.DWORD), ("parent", w.DWORD),
                    ("priority", w.LONG), ("flags", w.DWORD), ("name", w.WCHAR * 260)]
    class Counters(c.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [(name, c.c_size_t) for name in (
            "peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    kernel, psapi = c.WinDLL("kernel32", use_last_error=True), c.WinDLL("psapi", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [w.HANDLE, c.POINTER(Entry)]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Counters), w.DWORD]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    parents: dict[int, int] = {}
    entry = Entry()
    entry.dwSize = c.sizeof(entry)
    try:
        valid = kernel.Process32FirstW(snapshot, c.byref(entry))
        while valid:
            parents[int(entry.pid)] = int(entry.parent)
            valid = kernel.Process32NextW(snapshot, c.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    pids = {os.getpid()}
    while True:
        descendants = {pid for pid, parent in parents.items() if parent in pids}
        if descendants <= pids:
            break
        pids |= descendants
    total = 0
    for pid in pids:
        handle = kernel.OpenProcess(0x0410, False, pid)
        if handle:
            try:
                counters = Counters()
                counters.cb = c.sizeof(counters)
                if psapi.GetProcessMemoryInfo(handle, c.byref(counters), counters.cb):
                    total += int(counters.working)
            finally:
                kernel.CloseHandle(handle)
    return total


def _linux_rss() -> int:
    parents, sizes = {}, {}
    for path in Path("/proc").glob("[0-9]*"):
        try:
            fields = dict(line.split(":", 1) for line in (path / "status").read_text().splitlines() if ":" in line)
            parents[int(path.name)] = int(fields["PPid"])
            sizes[int(path.name)] = int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
        except (OSError, ValueError, KeyError):
            continue
    pids = {os.getpid()}
    while not {pid for pid, parent in parents.items() if parent in pids} <= pids:
        pids |= {pid for pid, parent in parents.items() if parent in pids}
    return sum(sizes.get(pid, 0) for pid in pids)


class PeakMemory:
    def __init__(self) -> None:
        self.peak_bytes = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while True:
            try:
                self.peak_bytes = max(self.peak_bytes, _windows_rss() if os.name == "nt" else _linux_rss())
            except (OSError, AttributeError):
                pass
            if self.stop_event.wait(0.1):
                break

    def __enter__(self) -> "PeakMemory":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
