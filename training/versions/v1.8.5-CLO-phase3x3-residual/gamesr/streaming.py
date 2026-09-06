from __future__ import annotations

import atexit
import ctypes
import os
import queue
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch


_TRUE = {"1", "true", "yes", "on", "enabled"}
_FALSE = {"0", "false", "no", "off", "disabled"}
_GIB = 1024 ** 3
_MIB = 1024 ** 2


def _normalize_mode(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value if value is not None else "auto").strip().lower()
    if text in _TRUE:
        return "on"
    if text in _FALSE:
        return "off"
    return "auto"


def path_is_remote(path: Path) -> bool:
    """Best-effort Windows remote-drive/UNC detection without touching the file."""
    text = os.fspath(path)
    if text.startswith("\\\\") or text.startswith("//"):
        return True
    if os.name != "nt":
        return False
    drive, _ = os.path.splitdrive(text)
    if not drive:
        return False
    try:
        # DRIVE_REMOTE == 4. This also catches mapped SMB drive letters.
        return int(ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")) == 4
    except Exception:
        return False


def dataset_has_remote_sources(dataset, sample_count: int = 16) -> bool:
    files = getattr(dataset, "files", None) or []
    if not files:
        return False
    count = min(max(1, int(sample_count)), len(files))
    if count == 1:
        probes = [files[0]]
    else:
        step = max(1, (len(files) - 1) // (count - 1))
        probes = [files[min(i * step, len(files) - 1)] for i in range(count)]
    return any(path_is_remote(Path(item)) for item in probes)


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_memory_bytes() -> tuple[int, int]:
    """Return (total_physical, currently_available_physical) bytes.

    Uses GlobalMemoryStatusEx on Windows, /proc/meminfo on Linux, and sysconf as
    a final fallback. No psutil dependency is required.
    """
    if os.name == "nt":
        try:
            status = _MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except Exception:
            pass

    try:
        meminfo = {}
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                meminfo[key] = int(raw.strip().split()[0]) * 1024
        total = int(meminfo.get("MemTotal", 0))
        available = int(meminfo.get("MemAvailable", meminfo.get("MemFree", 0)))
        if total > 0 and available >= 0:
            return total, available
    except Exception:
        pass

    try:
        page = int(os.sysconf("SC_PAGE_SIZE"))
        total = page * int(os.sysconf("SC_PHYS_PAGES"))
        available = page * int(os.sysconf("SC_AVPHYS_PAGES"))
        return total, available
    except Exception:
        return 0, 0


def memory_snapshot() -> dict:
    total, available = system_memory_bytes()
    return {
        "total_bytes": total,
        "available_bytes": available,
        "total_gib": round(total / _GIB, 2) if total else 0.0,
        "available_gib": round(available / _GIB, 2) if available else 0.0,
    }


def _parse_budget_bytes(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0, int(float(value) * _GIB))
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return None
    try:
        return max(0, int(float(text) * _GIB))
    except ValueError:
        return None


def choose_auto_cache_budget(settings: dict) -> dict:
    """Choose a conservative compressed-image RAM cache size from *current* RAM.

    Default policy:
      - keep roughly 8 GiB system headroom on a 32 GiB machine (scaled down on
        smaller systems),
      - use half of the remaining headroom,
      - never exceed 12 GiB or 35% of total physical RAM,
      - disable the persistent tier if less than 1 GiB is safely available.

    The cache stores compressed source bytes, not decoded float32 images.
    """
    total, available = system_memory_bytes()
    requested = _parse_budget_bytes(settings.get("cache_budget_gib", "auto"))
    reserve_cfg = max(0.0, float(settings.get("cache_reserve_gib", 8.0)))
    cap_cfg = max(0.0, float(settings.get("cache_cap_gib", 12.0)))
    fraction = min(0.90, max(0.05, float(settings.get("cache_auto_fraction", 0.50))))
    floor_gib = max(0.0, float(settings.get("cache_floor_gib", 1.0)))

    if requested is not None:
        budget = requested
        mode = "manual"
        effective_reserve = int(reserve_cfg * _GIB)
    elif total > 0 and available > 0:
        # On smaller machines, reserving a fixed 8 GiB can leave no cache at all.
        effective_reserve = min(int(reserve_cfg * _GIB), max(int(4 * _GIB), int(total * 0.25)))
        headroom = max(0, available - effective_reserve)
        total_cap = int(total * 0.35)
        configured_cap = int(cap_cfg * _GIB)
        budget = min(int(headroom * fraction), configured_cap, total_cap)
        if budget < int(floor_gib * _GIB):
            budget = 0
        mode = "auto"
    else:
        # If memory discovery fails, stay conservative rather than allocating a
        # large invisible cache. Streaming read-ahead still works without it.
        effective_reserve = int(reserve_cfg * _GIB)
        budget = 0
        mode = "auto-memory-unknown"

    return {
        "mode": mode,
        "total_bytes": total,
        "available_bytes": available,
        "total_gib": round(total / _GIB, 2) if total else 0.0,
        "available_gib": round(available / _GIB, 2) if available else 0.0,
        "budget_bytes": max(0, int(budget)),
        "budget_gib": round(max(0, int(budget)) / _GIB, 2),
        "reserve_gib": round(effective_reserve / _GIB, 2),
        "cap_gib": cap_cfg,
        "auto_fraction": fraction,
    }


def choose_validation_cache_budget(dataset, settings: dict, *, reserved_bytes: int = 0) -> dict:
    """Plan a bounded deterministic validation-tensor cache.

    Unlike the training cache, validation entries are already center-cropped and
    resized LR/HR float32 tensors.  They are much smaller than full screenshots
    and identical on every epoch, so caching them removes SMB reads *and* PIL
    decode/resize work after the first validation pass.

    ``reserved_bytes`` is the training cache budget.  It is subtracted even if
    the training cache has not filled yet, preventing two independent auto
    planners from overcommitting the same currently-free RAM.
    """
    total, available = system_memory_bytes()
    hr_patch = max(1, int(getattr(dataset, "hr_patch", 1)))
    lr_patch = max(1, int(getattr(dataset, "lr_patch", 1)))
    # Two float32 RGB tensors per sample (LR + HR). Python/container overhead is
    # covered by a 10% planning margin.
    raw_estimate = len(dataset) * 3 * (hr_patch * hr_patch + lr_patch * lr_patch) * 4
    estimated = int(raw_estimate * 1.10)

    requested = _parse_budget_bytes(settings.get("validation_cache_budget_gib", "auto"))
    reserve_gib = max(0.0, float(settings.get("validation_cache_reserve_gib", 7.0)))
    cap_gib = max(0.0, float(settings.get("validation_cache_cap_gib", 1.5)))
    fraction = min(0.90, max(0.05, float(settings.get("validation_cache_auto_fraction", 0.50))))
    floor_mib = max(0.0, float(settings.get("validation_cache_floor_mib", 64.0)))

    effective_reserve = int(reserve_gib * _GIB)
    if requested is not None:
        budget = min(requested, estimated)
        mode = "manual"
    elif total > 0 and available > 0:
        headroom = max(0, available - effective_reserve - max(0, int(reserved_bytes)))
        configured_cap = int(cap_gib * _GIB)
        total_cap = int(total * 0.08)
        budget = min(estimated, int(headroom * fraction), configured_cap, total_cap)
        if budget < int(floor_mib * _MIB):
            budget = 0
        mode = "auto"
    else:
        budget = 0
        mode = "auto-memory-unknown"

    return {
        "mode": mode,
        "total_bytes": total,
        "available_bytes": available,
        "total_gib": round(total / _GIB, 2) if total else 0.0,
        "available_gib": round(available / _GIB, 2) if available else 0.0,
        "reserved_training_bytes": max(0, int(reserved_bytes)),
        "reserved_training_gib": round(max(0, int(reserved_bytes)) / _GIB, 2),
        "estimated_bytes": estimated,
        "estimated_mib": round(estimated / _MIB, 1),
        "budget_bytes": max(0, int(budget)),
        "budget_mib": round(max(0, int(budget)) / _MIB, 1),
        "budget_gib": round(max(0, int(budget)) / _GIB, 2),
        "reserve_gib": reserve_gib,
        "cap_gib": cap_gib,
        "auto_fraction": fraction,
    }


class PersistentCompressedRamCache:
    """Bounded, cross-epoch cache for compressed source image bytes.

    The cache has two tiers:
      * protected tier (default 80%): once filled, it stays stable across epochs,
        avoiding LRU thrashing when the dataset is larger than RAM;
      * rolling LRU tier (default 20%): absorbs near-future/next-epoch prefetches.

    This gives a predictable persistent hit rate on large shuffled datasets while
    still allowing a small adaptive read-ahead area.
    """

    def __init__(
        self,
        budget_bytes: int,
        *,
        protected_ratio: float = 0.80,
        emergency_free_gib: float = 4.0,
        pressure_check_every: int = 32,
    ):
        self.budget_bytes = max(0, int(budget_bytes))
        self.protected_ratio = min(0.95, max(0.50, float(protected_ratio)))
        self.protected_budget = int(self.budget_bytes * self.protected_ratio)
        self.rolling_budget = max(0, self.budget_bytes - self.protected_budget)
        self.emergency_free_bytes = max(0, int(float(emergency_free_gib) * _GIB))
        self.pressure_check_every = max(1, int(pressure_check_every))

        self._protected: OrderedDict[int, bytes] = OrderedDict()
        self._rolling: OrderedDict[int, bytes] = OrderedDict()
        self._protected_bytes = 0
        self._rolling_bytes = 0
        self._lock = threading.RLock()
        self._puts_since_pressure_check = 0
        self._gets_since_pressure_check = 0
        self._evictions = 0
        self._pressure_evictions = 0
        self._too_large_skips = 0

    @property
    def enabled(self) -> bool:
        return self.budget_bytes > 0

    def get(self, key: int) -> bytes | None:
        if not self.enabled:
            return None
        with self._lock:
            payload = self._protected.get(key)
            if payload is None:
                payload = self._rolling.get(key)
                if payload is not None:
                    self._rolling.move_to_end(key)
            self._gets_since_pressure_check += 1
            # If another application consumes RAM after training starts, the
            # cache can shrink even during cache-hit-heavy later epochs.
            if self._gets_since_pressure_check >= max(64, self.pressure_check_every * 8):
                self._gets_since_pressure_check = 0
                self._pressure_trim_locked()
            return payload

    def _evict_rolling_until(self, max_bytes: int) -> None:
        while self._rolling and self._rolling_bytes > max_bytes:
            _, payload = self._rolling.popitem(last=False)
            self._rolling_bytes -= len(payload)
            self._evictions += 1

    def _pressure_trim_locked(self) -> None:
        if not self.emergency_free_bytes:
            return
        _, available = system_memory_bytes()
        if available <= 0 or available >= self.emergency_free_bytes:
            return

        # Avoid one Windows memory query per cached image. Release enough bytes
        # to cover the observed deficit plus a safety margin, rolling tier first.
        deficit = self.emergency_free_bytes - available
        target_release = max(deficit + 256 * _MIB, int((self._protected_bytes + self._rolling_bytes) * 0.10))
        released = 0
        while self._rolling and released < target_release:
            _, payload = self._rolling.popitem(last=False)
            size = len(payload)
            self._rolling_bytes -= size
            released += size
            self._evictions += 1
            self._pressure_evictions += 1
        while self._protected and released < target_release:
            _, payload = self._protected.popitem(last=False)
            size = len(payload)
            self._protected_bytes -= size
            released += size
            self._evictions += 1
            self._pressure_evictions += 1

    def put(self, key: int, payload: bytes) -> bool:
        if not self.enabled:
            return False
        size = len(payload)
        if size <= 0 or size > self.budget_bytes:
            with self._lock:
                self._too_large_skips += 1
            return False

        with self._lock:
            if key in self._protected or key in self._rolling:
                return True

            # Fill the stable tier first. It intentionally does not churn on every
            # miss once full; that is much better for shuffled full-dataset epochs.
            if self._protected_bytes + size <= self.protected_budget:
                self._protected[key] = payload
                self._protected_bytes += size
            elif self.rolling_budget > 0 and size <= self.rolling_budget:
                self._rolling[key] = payload
                self._rolling_bytes += size
                self._evict_rolling_until(self.rolling_budget)
            else:
                self._too_large_skips += 1
                return False

            self._puts_since_pressure_check += 1
            if self._puts_since_pressure_check >= self.pressure_check_every:
                self._puts_since_pressure_check = 0
                self._pressure_trim_locked()
            return True

    def snapshot(self) -> dict:
        total, available = system_memory_bytes()
        with self._lock:
            current = self._protected_bytes + self._rolling_bytes
            entries = len(self._protected) + len(self._rolling)
            return {
                "cache_enabled": self.enabled,
                "cache_budget_bytes": self.budget_bytes,
                "cache_budget_mib": round(self.budget_bytes / _MIB, 1),
                "cache_budget_gib": round(self.budget_bytes / _GIB, 2),
                "cache_bytes": current,
                "cache_mib": round(current / _MIB, 1),
                "cache_gib": round(current / _GIB, 2),
                "cache_entries": entries,
                "cache_protected_entries": len(self._protected),
                "cache_rolling_entries": len(self._rolling),
                "cache_protected_mib": round(self._protected_bytes / _MIB, 1),
                "cache_rolling_mib": round(self._rolling_bytes / _MIB, 1),
                "cache_evictions": self._evictions,
                "cache_pressure_evictions": self._pressure_evictions,
                "cache_too_large_skips": self._too_large_skips,
                "system_total_gib": round(total / _GIB, 2) if total else 0.0,
                "system_available_gib": round(available / _GIB, 2) if available else 0.0,
            }


@dataclass
class _StreamingStats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    decode_service_seconds: float = 0.0
    consumed_samples: int = 0
    ready_samples: int = 0
    batch_build_seconds: float = 0.0
    batches: int = 0
    ready_queue_sum: int = 0
    ready_queue_samples: int = 0

    def add_decode(self, seconds: float) -> None:
        with self.lock:
            self.decode_service_seconds += float(seconds)

    def add_batch(self, consumed: int, ready: int, build_seconds: float, ready_queue: int) -> None:
        with self.lock:
            self.consumed_samples += int(consumed)
            self.ready_samples += int(ready)
            self.batch_build_seconds += float(build_seconds)
            self.batches += 1
            self.ready_queue_sum += int(ready_queue)
            self.ready_queue_samples += 1

    def snapshot(self) -> dict:
        with self.lock:
            consumed = max(1, self.consumed_samples)
            batches = max(1, self.batches)
            queue_n = max(1, self.ready_queue_samples)
            return {
                "decode_service_seconds": round(self.decode_service_seconds, 3),
                "batch_build_seconds": round(self.batch_build_seconds, 3),
                "avg_batch_build_ms": round(1000.0 * self.batch_build_seconds / batches, 2),
                "prefetch_hit_pct": round(100.0 * self.ready_samples / consumed, 1),
                "avg_ready_queue": round(self.ready_queue_sum / queue_n, 1),
                "consumed_samples": self.consumed_samples,
            }


class RamStreamingBatchLoader:
    """Persistent remote-data pipeline for repeated GameSR epochs.

    v1.7.5 keeps compressed source bytes in a bounded RAM cache *across epochs*,
    keeps the SMB/decode thread pools alive, and prewarms the beginning of the
    next epoch while validation/checkpoint work is happening.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        read_ahead_samples: int = 256,
        fetch_threads: int = 24,
        decode_threads: int = 6,
        pin_memory: bool = True,
        cache_budget_gib="auto",
        cache_reserve_gib: float = 8.0,
        cache_cap_gib: float = 12.0,
        cache_auto_fraction: float = 0.50,
        cache_floor_gib: float = 1.0,
        cache_protected_ratio: float = 0.80,
        emergency_free_gib: float = 4.0,
        next_epoch_prefetch_samples: int = 256,
    ):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.read_ahead_samples = max(self.batch_size * 2, int(read_ahead_samples))
        self.fetch_threads = max(1, int(fetch_threads))
        self.decode_threads = max(1, int(decode_threads))
        self.pin_memory = bool(pin_memory)
        self.next_epoch_prefetch_samples = max(0, int(next_epoch_prefetch_samples))
        self._epoch = 0
        self._latest_stats: dict = {}
        self._closed = False

        plan_settings = {
            "cache_budget_gib": cache_budget_gib,
            "cache_reserve_gib": cache_reserve_gib,
            "cache_cap_gib": cache_cap_gib,
            "cache_auto_fraction": cache_auto_fraction,
            "cache_floor_gib": cache_floor_gib,
        }
        self.cache_plan = choose_auto_cache_budget(plan_settings)
        self._cache = PersistentCompressedRamCache(
            self.cache_plan["budget_bytes"],
            protected_ratio=float(cache_protected_ratio),
            emergency_free_gib=float(emergency_free_gib),
        )

        self._fetch_pool: ThreadPoolExecutor | None = None
        self._decode_pool: ThreadPoolExecutor | None = None
        self._inflight: dict[int, Future] = {}
        self._inflight_lock = threading.Lock()

        self._service_lock = threading.Lock()
        self._network_bytes = 0
        self._network_samples = 0
        self._network_seconds = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._inflight_reuse = 0
        self._next_epoch_submitted = 0

        atexit.register(self.close)

    def _ensure_pools(self) -> None:
        if self._fetch_pool is None:
            self._fetch_pool = ThreadPoolExecutor(max_workers=self.fetch_threads, thread_name_prefix="gamesr-smb")
        if self._decode_pool is None:
            self._decode_pool = ThreadPoolExecutor(max_workers=self.decode_threads, thread_name_prefix="gamesr-decode")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._decode_pool is not None:
            self._decode_pool.shutdown(wait=False, cancel_futures=True)
            self._decode_pool = None
        if self._fetch_pool is not None:
            self._fetch_pool.shutdown(wait=False, cancel_futures=True)
            self._fetch_pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _service_snapshot(self) -> dict:
        with self._service_lock:
            return {
                "network_bytes": self._network_bytes,
                "network_samples": self._network_samples,
                "network_seconds": self._network_seconds,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "inflight_reuse": self._inflight_reuse,
            }

    @staticmethod
    def _delta(after: dict, before: dict, key: str):
        return after.get(key, 0) - before.get(key, 0)

    def stats_snapshot(self) -> dict:
        return dict(self._latest_stats)

    def _completed_future(self, payload: bytes) -> Future:
        future = Future()
        future.set_result(payload)
        return future

    def _fetch_network(self, dataset_index: int) -> bytes:
        start = time.perf_counter()
        payload = self.dataset.read_source_bytes(dataset_index)
        elapsed = time.perf_counter() - start
        self._cache.put(dataset_index, payload)
        with self._service_lock:
            self._network_bytes += len(payload)
            self._network_samples += 1
            self._network_seconds += elapsed
        return payload

    def _request_raw(self, dataset_index: int) -> Future:
        payload = self._cache.get(dataset_index)
        if payload is not None:
            with self._service_lock:
                self._cache_hits += 1
            return self._completed_future(payload)

        with self._service_lock:
            self._cache_misses += 1

        self._ensure_pools()
        assert self._fetch_pool is not None
        with self._inflight_lock:
            existing = self._inflight.get(dataset_index)
            if existing is not None:
                with self._service_lock:
                    self._inflight_reuse += 1
                return existing
            future = self._fetch_pool.submit(self._fetch_network, dataset_index)
            self._inflight[dataset_index] = future

        # Register the callback *after* releasing _inflight_lock. A very fast
        # local/test read can finish before add_done_callback(); Future then runs
        # the callback synchronously in this thread, which would deadlock if the
        # lock were still held.
        def clear(done: Future, key: int = dataset_index) -> None:
            with self._inflight_lock:
                if self._inflight.get(key) is done:
                    self._inflight.pop(key, None)

        future.add_done_callback(clear)
        return future

    def _order_for_epoch(self, epoch: int) -> list[int]:
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + epoch * 1_000_003).shuffle(order)
        if self.drop_last:
            keep = (len(order) // self.batch_size) * self.batch_size
            order = order[:keep]
        return order

    def _schedule_next_epoch_prewarm(self, next_epoch: int) -> int:
        if self.next_epoch_prefetch_samples <= 0 or self._closed:
            return 0
        order = self._order_for_epoch(next_epoch)
        count = min(len(order), self.next_epoch_prefetch_samples)
        # Requests are intentionally fire-and-forget. They continue while the
        # current epoch is validating/saving its checkpoint.
        for dataset_index in order[:count]:
            self._request_raw(dataset_index)
        self._next_epoch_submitted = count
        return count

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self._closed:
            raise RuntimeError("RamStreamingBatchLoader is closed")
        self._ensure_pools()
        assert self._decode_pool is not None

        self._epoch += 1
        epoch = self._epoch
        stats = _StreamingStats()
        service_before = self._service_snapshot()
        memory_before = memory_snapshot()
        order = self._order_for_epoch(epoch)

        raw_futures: dict[int, Future] = {}
        decode_futures: dict[int, Future] = {}
        submit_cursor = 0
        consume_cursor = 0
        completed = False

        def decode_task(position: int, raw_future: Future):
            payload = raw_future.result()
            start = time.perf_counter()
            rng = random.Random(self.seed * 1009 + epoch * 1_000_003 + position * 9176)
            tensor = self.dataset.decode_hr_bytes(payload, rng=rng)
            stats.add_decode(time.perf_counter() - start)
            return tensor

        def fill_window() -> None:
            nonlocal submit_cursor
            target = min(len(order), consume_cursor + self.read_ahead_samples)
            while submit_cursor < target:
                pos = submit_cursor
                dataset_index = order[pos]
                raw = self._request_raw(dataset_index)
                raw_futures[pos] = raw
                decode_futures[pos] = self._decode_pool.submit(decode_task, pos, raw)
                submit_cursor += 1

        try:
            fill_window()
            while consume_cursor < len(order):
                batch_start = time.perf_counter()
                end = min(len(order), consume_cursor + self.batch_size)
                if self.drop_last and end - consume_cursor < self.batch_size:
                    break

                positions = list(range(consume_cursor, end))
                ready_before_wait = sum(1 for pos in positions if decode_futures[pos].done())
                ready_queue = sum(1 for pos in range(consume_cursor, submit_cursor) if decode_futures[pos].done())
                tensors = [decode_futures[pos].result() for pos in positions]
                batch = torch.stack(tensors, dim=0).contiguous()
                if self.pin_memory and torch.cuda.is_available():
                    try:
                        batch = batch.pin_memory()
                    except RuntimeError:
                        pass

                for pos in positions:
                    raw_futures.pop(pos, None)
                    decode_futures.pop(pos, None)
                consume_cursor = end
                fill_window()

                stats.add_batch(
                    consumed=len(positions),
                    ready=ready_before_wait,
                    build_seconds=time.perf_counter() - batch_start,
                    ready_queue=ready_queue,
                )
                yield batch
            completed = consume_cursor >= len(order)
        finally:
            service_after = self._service_snapshot()
            cache = self._cache.snapshot()
            memory_after = memory_snapshot()
            stats_payload = stats.snapshot()
            hits = max(0, int(self._delta(service_after, service_before, "cache_hits")))
            misses = max(0, int(self._delta(service_after, service_before, "cache_misses")))
            hit_denom = max(1, hits + misses)
            network_bytes = max(0, int(self._delta(service_after, service_before, "network_bytes")))
            network_samples = max(0, int(self._delta(service_after, service_before, "network_samples")))
            network_seconds = max(0.0, float(self._delta(service_after, service_before, "network_seconds")))
            inflight_reuse = max(0, int(self._delta(service_after, service_before, "inflight_reuse")))

            self._latest_stats = {
                "enabled": True,
                "mode": "persistent_ram_streaming_v175",
                "epoch": epoch,
                "read_ahead_samples": self.read_ahead_samples,
                "fetch_threads": self.fetch_threads,
                "decode_threads": self.decode_threads,
                "pin_memory": self.pin_memory,
                "bytes_read": network_bytes,
                "read_mib": round(network_bytes / _MIB, 1),
                "network_samples": network_samples,
                "network_service_seconds": round(network_seconds, 3),
                "network_service_mib_s": round((network_bytes / _MIB) / max(network_seconds, 1e-9), 1) if network_bytes else 0.0,
                "cache_hits": hits,
                "cache_misses": misses,
                "cache_hit_pct": round(100.0 * hits / hit_denom, 1),
                "inflight_reuse": inflight_reuse,
                "memory_available_before_gib": memory_before.get("available_gib", 0.0),
                "memory_available_after_gib": memory_after.get("available_gib", 0.0),
                "next_epoch_prefetch_samples": self.next_epoch_prefetch_samples,
                **stats_payload,
                **cache,
            }

            if completed:
                self._latest_stats["next_epoch_prewarm_submitted"] = self._schedule_next_epoch_prewarm(epoch + 1)
            else:
                self._latest_stats["next_epoch_prewarm_submitted"] = 0


class PersistentValidationTensorCache:
    """Stable bounded cache for deterministic validation LR/HR CPU tensors.

    The cache intentionally fills once and does not LRU-thrash on a sequential
    validation scan. If a future validation set is larger than the budget, the
    cached prefix remains useful on every later epoch instead of being evicted by
    the remainder of the same scan.
    """

    def __init__(self, budget_bytes: int, *, emergency_free_gib: float = 4.0):
        self.budget_bytes = max(0, int(budget_bytes))
        self.emergency_free_bytes = max(0, int(float(emergency_free_gib) * _GIB))
        self._items: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self._pressure_evictions = 0
        self._full_skips = 0
        self._ops = 0

    @property
    def enabled(self) -> bool:
        return self.budget_bytes > 0

    @staticmethod
    def _pair_bytes(pair: tuple[torch.Tensor, torch.Tensor]) -> int:
        return sum(int(t.numel()) * int(t.element_size()) for t in pair)

    def _pressure_trim_locked(self) -> None:
        if not self.emergency_free_bytes or not self._items:
            return
        _, available = system_memory_bytes()
        if available <= 0 or available >= self.emergency_free_bytes:
            return
        deficit = self.emergency_free_bytes - available
        release_target = max(deficit + 128 * _MIB, int(self._bytes * 0.15))
        released = 0
        # Drop newest entries first so the stable prefix remains useful.
        while self._items and released < release_target:
            _, pair = self._items.popitem(last=True)
            size = self._pair_bytes(pair)
            self._bytes -= size
            released += size
            self._pressure_evictions += 1

    def get(self, key: int):
        if not self.enabled:
            return None
        with self._lock:
            pair = self._items.get(key)
            self._ops += 1
            if self._ops % 256 == 0:
                self._pressure_trim_locked()
            return pair

    def put(self, key: int, pair: tuple[torch.Tensor, torch.Tensor]) -> bool:
        if not self.enabled:
            return False
        size = self._pair_bytes(pair)
        if size <= 0 or size > self.budget_bytes:
            self._full_skips += 1
            return False
        with self._lock:
            if key in self._items:
                return True
            if self._bytes + size > self.budget_bytes:
                self._full_skips += 1
                return False
            self._items[key] = pair
            self._bytes += size
            self._ops += 1
            if self._ops % 64 == 0:
                self._pressure_trim_locked()
            return True

    def snapshot(self) -> dict:
        total, available = system_memory_bytes()
        with self._lock:
            return {
                "val_cache_enabled": self.enabled,
                "val_cache_budget_bytes": self.budget_bytes,
                "val_cache_budget_mib": round(self.budget_bytes / _MIB, 1),
                "val_cache_bytes": self._bytes,
                "val_cache_mib": round(self._bytes / _MIB, 1),
                "val_cache_entries": len(self._items),
                "val_cache_pressure_evictions": self._pressure_evictions,
                "val_cache_full_skips": self._full_skips,
                "val_system_total_gib": round(total / _GIB, 2) if total else 0.0,
                "val_system_available_gib": round(available / _GIB, 2) if available else 0.0,
            }


class PersistentValidationLoader:
    """Remote validation loader with cross-epoch processed-tensor RAM cache."""

    def __init__(
        self,
        dataset,
        *,
        batch_size: int = 1,
        read_ahead_samples: int = 96,
        threads: int = 12,
        pin_memory: bool = True,
        cache_budget_gib="auto",
        cache_reserve_gib: float = 7.0,
        cache_cap_gib: float = 1.5,
        cache_auto_fraction: float = 0.50,
        cache_floor_mib: float = 64.0,
        emergency_free_gib: float = 4.0,
        reserved_training_bytes: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.read_ahead_samples = max(self.batch_size * 2, int(read_ahead_samples))
        self.threads = max(1, int(threads))
        self.pin_memory = bool(pin_memory)
        self._pool: ThreadPoolExecutor | None = None
        self._closed = False
        self._latest_stats: dict = {}
        self._lock = threading.Lock()
        self._network_bytes = 0
        self._network_seconds = 0.0
        self._decode_seconds = 0.0
        self._cache_hits = 0
        self._cache_misses = 0

        plan_settings = {
            "validation_cache_budget_gib": cache_budget_gib,
            "validation_cache_reserve_gib": cache_reserve_gib,
            "validation_cache_cap_gib": cache_cap_gib,
            "validation_cache_auto_fraction": cache_auto_fraction,
            "validation_cache_floor_mib": cache_floor_mib,
        }
        self.cache_plan = choose_validation_cache_budget(
            dataset, plan_settings, reserved_bytes=reserved_training_bytes
        )
        self._cache = PersistentValidationTensorCache(
            self.cache_plan["budget_bytes"], emergency_free_gib=emergency_free_gib
        )
        atexit.register(self.close)

    def _ensure_pool(self) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="gamesr-val")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _snapshot_service(self) -> dict:
        with self._lock:
            return {
                "network_bytes": self._network_bytes,
                "network_seconds": self._network_seconds,
                "decode_seconds": self._decode_seconds,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }

    def stats_snapshot(self) -> dict:
        return dict(self._latest_stats)

    def _load_pair(self, index: int):
        pair = self._cache.get(index)
        if pair is not None:
            with self._lock:
                self._cache_hits += 1
            return pair

        with self._lock:
            self._cache_misses += 1
        net_start = time.perf_counter()
        payload = self.dataset.read_source_bytes(index)
        net_seconds = time.perf_counter() - net_start
        decode_start = time.perf_counter()
        pair = self.dataset.decode_validation_pair_bytes(payload)
        decode_seconds = time.perf_counter() - decode_start
        if self.pin_memory and torch.cuda.is_available():
            try:
                pair = (pair[0].pin_memory(), pair[1].pin_memory())
            except RuntimeError:
                pass
        self._cache.put(index, pair)
        with self._lock:
            self._network_bytes += len(payload)
            self._network_seconds += net_seconds
            self._decode_seconds += decode_seconds
        return pair

    def __iter__(self):
        if self._closed:
            raise RuntimeError("PersistentValidationLoader is closed")
        self._ensure_pool()
        assert self._pool is not None
        before = self._snapshot_service()
        started = time.perf_counter()
        futures: dict[int, Future] = {}
        submit = 0
        consume = 0
        n = len(self.dataset)

        def fill() -> None:
            nonlocal submit
            target = min(n, consume + self.read_ahead_samples)
            while submit < target:
                futures[submit] = self._pool.submit(self._load_pair, submit)
                submit += 1

        try:
            fill()
            while consume < n:
                end = min(n, consume + self.batch_size)
                pairs = [futures[i].result() for i in range(consume, end)]
                for i in range(consume, end):
                    futures.pop(i, None)
                consume = end
                fill()
                lr = torch.stack([p[0] for p in pairs], dim=0).contiguous()
                hr = torch.stack([p[1] for p in pairs], dim=0).contiguous()
                yield lr, hr
        finally:
            after = self._snapshot_service()
            elapsed = time.perf_counter() - started
            net_bytes = max(0, after["network_bytes"] - before["network_bytes"])
            net_seconds = max(0.0, after["network_seconds"] - before["network_seconds"])
            decode_seconds = max(0.0, after["decode_seconds"] - before["decode_seconds"])
            hits = max(0, after["cache_hits"] - before["cache_hits"])
            misses = max(0, after["cache_misses"] - before["cache_misses"])
            denom = max(1, hits + misses)
            self._latest_stats = {
                "enabled": True,
                "mode": "persistent_validation_tensor_cache_v176",
                "elapsed_seconds": round(elapsed, 3),
                "read_mib": round(net_bytes / _MIB, 1),
                "network_service_seconds": round(net_seconds, 3),
                "network_service_mib_s": round((net_bytes / _MIB) / max(net_seconds, 1e-9), 1) if net_bytes else 0.0,
                "decode_service_seconds": round(decode_seconds, 3),
                "cache_hits": hits,
                "cache_misses": misses,
                "cache_hit_pct": round(100.0 * hits / denom, 1),
                "read_ahead_samples": self.read_ahead_samples,
                "threads": self.threads,
                **self._cache.snapshot(),
            }


class BackgroundBatchLoader:
    """Assemble/pin several CPU batches ahead of the training loop."""

    def __init__(self, loader, queue_batches: int = 4):
        self.loader = loader
        self.dataset = loader.dataset
        self.queue_batches = max(1, int(queue_batches))

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def cache_plan(self) -> dict:
        return dict(getattr(self.loader, "cache_plan", {}) or {})

    def stats_snapshot(self) -> dict:
        getter = getattr(self.loader, "stats_snapshot", None)
        payload = getter() if getter else {}
        payload = dict(payload)
        payload["batch_queue"] = self.queue_batches
        return payload

    def close(self) -> None:
        close = getattr(self.loader, "close", None)
        if close:
            close()

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self.queue_batches)
        sentinel = object()
        stop = threading.Event()

        def put_item(item) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def producer() -> None:
            try:
                for batch in self.loader:
                    if not put_item((True, batch)):
                        return
            except BaseException as exc:
                put_item((False, exc))
            finally:
                put_item((True, sentinel))

        thread = threading.Thread(target=producer, name="gamesr-batch-queue", daemon=True)
        thread.start()
        try:
            while True:
                ok, item = q.get()
                if ok and item is sentinel:
                    break
                if not ok:
                    raise item
                yield item
        finally:
            stop.set()


class CudaPrefetchLoader:
    """Double-buffer CPU batches onto a dedicated CUDA copy stream."""

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.dataset = loader.dataset
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def cache_plan(self) -> dict:
        return dict(getattr(self.loader, "cache_plan", {}) or {})

    def stats_snapshot(self) -> dict:
        getter = getattr(self.loader, "stats_snapshot", None)
        payload = getter() if getter else {}
        payload = dict(payload)
        payload["cuda_prefetch"] = True
        return payload

    def close(self) -> None:
        close = getattr(self.loader, "close", None)
        if close:
            close()

    def __iter__(self):
        iterator = iter(self.loader)
        copy_stream = torch.cuda.Stream(device=self.device)

        try:
            first_cpu = next(iterator)
        except StopIteration:
            return

        def copy_async(batch: torch.Tensor) -> torch.Tensor:
            with torch.cuda.stream(copy_stream):
                return batch.to(self.device, non_blocking=True)

        next_gpu = copy_async(first_cpu)
        while next_gpu is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(copy_stream)
            current = next_gpu
            current.record_stream(current_stream)

            try:
                next_cpu = next(iterator)
            except StopIteration:
                next_gpu = None
            else:
                next_gpu = copy_async(next_cpu)
            yield current


def streaming_settings(data_cfg: dict) -> dict:
    cfg = dict(data_cfg.get("streaming_prefetch") or {})
    env_mode = os.environ.get("GAMESR_STREAMING_PREFETCH")
    if env_mode is not None:
        cfg["enabled"] = env_mode
    if "GAMESR_READ_AHEAD" in os.environ:
        cfg["read_ahead_samples"] = int(os.environ["GAMESR_READ_AHEAD"])
    if "GAMESR_FETCH_THREADS" in os.environ:
        cfg["fetch_threads"] = int(os.environ["GAMESR_FETCH_THREADS"])
    if "GAMESR_DECODE_THREADS" in os.environ:
        cfg["decode_threads"] = int(os.environ["GAMESR_DECODE_THREADS"])
    if "GAMESR_RAM_CACHE_GIB" in os.environ:
        cfg["cache_budget_gib"] = os.environ["GAMESR_RAM_CACHE_GIB"]
    if "GAMESR_RAM_CACHE_CAP_GIB" in os.environ:
        cfg["cache_cap_gib"] = float(os.environ["GAMESR_RAM_CACHE_CAP_GIB"])
    if "GAMESR_RAM_CACHE_RESERVE_GIB" in os.environ:
        cfg["cache_reserve_gib"] = float(os.environ["GAMESR_RAM_CACHE_RESERVE_GIB"])
    if "GAMESR_VAL_RAM_CACHE_GIB" in os.environ:
        cfg["validation_cache_budget_gib"] = os.environ["GAMESR_VAL_RAM_CACHE_GIB"]
    if "GAMESR_VAL_CACHE" in os.environ:
        cfg["validation_cache"] = os.environ["GAMESR_VAL_CACHE"]

    cfg.setdefault("enabled", "auto")
    cfg.setdefault("read_ahead_samples", 256)
    cfg.setdefault("fetch_threads", 24)
    cfg.setdefault("decode_threads", max(2, min(8, int(data_cfg.get("workers", 6) or 6))))
    cfg.setdefault("batch_queue", 4)
    cfg.setdefault("cuda_prefetch", True)
    cfg.setdefault("pin_memory", bool(data_cfg.get("pin_memory", True)))

    # v1.7.5 persistent compressed-byte cache. 'auto' is calculated from the
    # amount of RAM currently available when training starts.
    cfg.setdefault("persistent_cache", True)
    cfg.setdefault("cache_budget_gib", "auto")
    cfg.setdefault("cache_reserve_gib", 8.0)
    cfg.setdefault("cache_cap_gib", 12.0)
    cfg.setdefault("cache_auto_fraction", 0.50)
    cfg.setdefault("cache_floor_gib", 1.0)
    cfg.setdefault("cache_protected_ratio", 0.80)
    cfg.setdefault("emergency_free_gib", 4.0)
    cfg.setdefault("next_epoch_prefetch_samples", 256)

    # v1.7.6 deterministic validation cache. It stores already center-cropped
    # float32 LR/HR tensors, so later validation epochs skip SMB + PIL entirely.
    cfg.setdefault("validation_cache", "auto")
    cfg.setdefault("validation_cache_budget_gib", "auto")
    cfg.setdefault("validation_cache_reserve_gib", 7.0)
    cfg.setdefault("validation_cache_cap_gib", 1.5)
    cfg.setdefault("validation_cache_auto_fraction", 0.50)
    cfg.setdefault("validation_cache_floor_mib", 64.0)
    cfg.setdefault("validation_read_ahead_samples", 96)
    cfg.setdefault("validation_threads", 12)
    cfg.setdefault("validation_pin_memory", True)

    cfg["mode"] = _normalize_mode(cfg.get("enabled"))
    cfg["validation_mode"] = _normalize_mode(cfg.get("validation_cache"))
    if not bool(cfg.get("persistent_cache", True)):
        cfg["cache_budget_gib"] = 0
    return cfg


def should_use_streaming(dataset, settings: dict) -> tuple[bool, bool]:
    remote = dataset_has_remote_sources(dataset)
    mode = settings.get("mode", "auto")
    if mode == "on":
        return True, remote
    if mode == "off":
        return False, remote
    return remote, remote


def should_use_validation_cache(dataset, settings: dict) -> tuple[bool, bool]:
    remote = dataset_has_remote_sources(dataset)
    mode = settings.get("validation_mode", "auto")
    if mode == "on":
        return True, remote
    if mode == "off":
        return False, remote
    return remote, remote
