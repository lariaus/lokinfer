"""Remote executor client: run compiled models on another device (the iPhone
test host, or a local `lokinfer-serve`). Protocol: include/lokinfer/remote/server.h.

    dev = RemoteDevice.connect("<udid>.coredevice.local")
    mid = dev.load(model)               # CompiledModel or .lkm path; cached by sha256
    outs, times_ns = dev.run(mid, [x, y], iters=10)

Test data crosses the cable once: `dev.put_tensor(x)` stores an input / reference by sha256 (the phone keeps it
across sessions), `dev.run2(mid, [Blob(sha), Generated(...), x], compare=[...])` runs on blobs, on inputs generated
on the device from a seed, and compares outputs on the device (against reference blobs or other outputs).

Device tensors (feature "device_tensors") stay on the device across calls: `kv = dev.alloc_tensor(np.float16, shape)`
(zeroed), `dev.run2(mid, [ids, pos, kv])` binds it in place (no copy, also on the GPU), `dev.read_tensor(kv)` /
`dev.write_tensor(kv, a)` transfer it only when asked, `dev.free_tensor(kv)`. A model's mutable inputs
(lk.MutableType, e.g. a KV cache) must be device tensors: they are written in place, on the device.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

from . import dtypes as _dt

MAGIC = 0x494B4F4C
VERSION = 1
HELLO, HAS_BLOB, PUT_BLOB, LOAD_MODEL, RUN, SHUTDOWN = 1, 2, 3, 4, 5, 6
LOAD_MODEL_SPLIT = 16  # optional: feature "split_models"
BLOBS = 17  # optional: feature "blobs"
FREE_MODEL = 18  # optional: feature "free_model"
PIN_MODEL = 19  # optional: feature "pin_model"
PROFILE, TAKE_PROFILE, GPU_INSPECT = 7, 8, 9  # optional: advertised in HELLO's "features"
MEMORY, RUN2 = 10, 11
ALLOC_TENSOR, WRITE_TENSOR, READ_TENSOR, FREE_TENSOR = 12, 13, 14, 15  # feature "device_tensors"
DEFAULT_PORT = 47800
_HDR = struct.Struct("<IHHIIQ")
# IR DType enum order (include/lokinfer/ir/types.h)
_DTYPES = [_dt.f32, _dt.f16, _dt.bf16, _dt.i8, _dt.u8, _dt.i16, _dt.i32, _dt.i64, _dt.bool_]


class RemoteError(RuntimeError):
    pass


class Blob(str):
    """A RUN2 input / reference given by the sha256 of a tensor blob already on the device (`put_tensor`)."""


class DeviceTensor:
    """A tensor living on the device (`RemoteDevice.alloc_tensor`): a RUN2 input bound in place, kept across calls."""

    def __init__(self, handle: int, dtype, shape):
        self.handle = int(handle)
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(d) for d in shape)

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    def __repr__(self):
        return f"DeviceTensor(#{self.handle}, {self.dtype}{list(self.shape)})"


class Generated:
    """A RUN2 input generated on the device from a seed (nothing crosses the cable): dist "normal" (mean a, std b),
    "uniform" [a, b) or "int" [a, b). `generate()` reproduces it on the Mac."""

    DISTS = {"normal": 0, "uniform": 1, "int": 2}

    def __init__(self, shape, seed: int, dtype=np.float32, dist: str = "normal", a: float = 0.0, b: float = 1.0):
        self.shape = tuple(int(d) for d in shape)
        self.seed = int(seed) & (2**64 - 1)
        self.dtype = np.dtype(dtype)
        self.dist = dist
        self.a, self.b = float(a), float(b)
        if self.dist not in self.DISTS:
            raise ValueError(f"unknown dist {dist!r}")

    def encode(self) -> bytes:
        dt = _DTYPES.index(_dt.from_numpy_dtype(self.dtype))
        return (struct.pack("<II", dt, len(self.shape)) + struct.pack(f"<{len(self.shape)}q", *self.shape)
                + struct.pack("<QIdd", self.seed, self.DISTS[self.dist], self.a, self.b))

    def numpy(self) -> np.ndarray:
        return generate(self.shape, self.seed, self.dtype, self.dist, self.a, self.b)


def _splitmix(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def generate(shape, seed: int, dtype=np.float32, dist: str = "normal", a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """The device's RUN2 input generator (src/remote/server.cpp `generateTensor`), in numpy. f32 normals can differ
    from the device's by an ulp where libm's log/cos round differently."""
    n = int(np.prod(shape, dtype=np.int64)) if len(shape) else 1
    i = np.arange(n, dtype=np.uint64)
    golden = np.uint64(0x9E3779B97F4A7C15)
    seed = np.uint64(int(seed) & (2**64 - 1))
    with np.errstate(over="ignore"):
        u1 = (_splitmix(seed + (np.uint64(2) * i + np.uint64(1)) * golden) >> np.uint64(11)).astype(np.float64) * 2.0**-53
        if dist == "normal":
            u2 = (_splitmix(seed + (np.uint64(2) * i + np.uint64(2)) * golden) >> np.uint64(11)).astype(np.float64) * 2.0**-53
            x = a + b * np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(6.283185307179586 * u2)
        elif dist == "uniform":
            x = a + (b - a) * u1
        else:
            x = np.floor(a) + np.floor(u1 * (np.floor(b) - np.floor(a)))
    return x.astype(dtype).reshape(shape)


class Stats:
    """On-device comparison of one output (RUN2): mismatches = #|got - ref| > atol + rtol |ref|."""

    def __init__(self, output, count, nonfinite, mismatches, worst, max_abs_err, max_abs_ref, sse, ssr):
        self.output, self.count, self.nonfinite, self.mismatches, self.worst = output, count, nonfinite, mismatches, worst
        self.max_abs_err, self.max_abs_ref, self.sum_sq_err, self.sum_sq_ref = max_abs_err, max_abs_ref, sse, ssr

    @property
    def ok(self) -> bool:
        return self.mismatches == 0 and self.nonfinite == 0

    @property
    def rel_rms(self) -> float:
        return float(np.sqrt(self.sum_sq_err / self.sum_sq_ref)) if self.sum_sq_ref > 0 else float(np.sqrt(self.sum_sq_err))

    @property
    def snr_db(self) -> float:
        return float(10 * np.log10(self.sum_sq_ref / self.sum_sq_err)) if self.sum_sq_err > 0 else float("inf")

    def __repr__(self):
        return (f"Stats(out {self.output}: {self.mismatches}/{self.count} mismatches, nonfinite {self.nonfinite}, "
                f"max err {self.max_abs_err:.3g} (max ref {self.max_abs_ref:.3g}), rel rms {self.rel_rms:.3g}, worst @{self.worst})")


def _encode_tensor(x) -> bytes:
    if isinstance(x, (int, float)):
        raise TypeError("pass scalars as 0-d numpy arrays with the right dtype")
    a = np.asarray(x)
    a = a if a.flags.c_contiguous else a.copy(order="C")  # not ascontiguousarray: it turns 0-d into 1-d
    dt = _dt.from_numpy_dtype(a.dtype)
    head = struct.pack("<II", _DTYPES.index(dt), a.ndim) + struct.pack(f"<{a.ndim}q", *a.shape)
    raw = a.tobytes()
    return head + struct.pack("<Q", len(raw)) + raw


class RemoteDevice:
    def __init__(self, sock: socket.socket):
        # The user took the phone away (see scripts/ios_deploy.sh). Local executors (lokinfer-serve on loopback, the
        # Mac-side protocol tests) stay allowed.
        try:
            local = sock.getpeername()[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")
        except OSError:
            local = False
        if os.path.exists("/tmp/lokinfer-phone-disabled") and not local:
            sock.close()
            raise RemoteError("iPhone disabled by the user: " + open("/tmp/lokinfer-phone-disabled").read().strip())
        self._sock = sock
        self._next_id = 1
        self.info = json.loads(self._call(HELLO, b"").decode() or "{}")

    @classmethod
    def connect(cls, host: str, port: int = DEFAULT_PORT, timeout: float = 30.0, retries: int = 20) -> "RemoteDevice":
        last = None
        for attempt in range(retries):
            try:
                s = socket.create_connection((host, port), timeout=timeout)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                dev = cls(s)
                dev._addr = (host, port, timeout)
                return dev
            except OSError as e:  # tunnel not up yet / app starting
                last = e
                time.sleep(min(0.25 * (attempt + 1), 2.0))
        raise RemoteError(f"cannot connect to {host}:{port}: {last}")

    @property
    def broken(self) -> bool:
        """True after a failed call (the connection can't be reused: reconnect())."""
        return bool(self.__dict__.get("_broken"))

    def reconnect(self, retries: int = 120) -> None:
        """A new connection to the same server, in place (the same object stays valid; `session` is incremented). The
        server freed everything the old connection owned (models, device tensors): ids from before are gone. For a
        LokiHost that crashed, the deploy watchdog relaunches it within seconds; this waits for it (~1 min max)."""
        host, port, timeout = self._addr
        try:
            self._sock.close()
        except OSError:
            pass
        fresh = type(self).connect(host, port, timeout=timeout, retries=retries)
        self._sock, self._next_id, self.info = fresh._sock, fresh._next_id, fresh.info
        self.__dict__.pop("_broken", None)
        self.__dict__.pop("_known_blobs", None)
        self.session = self.__dict__.get("session", 0) + 1

    def state(self) -> dict:
        """Fresh device info (the iPhone host reports live thermal state and Low Power Mode)."""
        self.info = json.loads(self._call(HELLO, b"").decode() or "{}")
        return self.info

    def close(self) -> None:
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- framing
    def _recv(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                raise RemoteError("connection closed")
            buf += chunk
        return bytes(buf)

    def _call(self, mtype: int, payload: bytes) -> bytes:
        if self.__dict__.get("_broken"):
            raise RemoteError(f"connection unusable after an earlier failure: {self._broken}")
        rid = self._next_id
        self._next_id += 1
        try:
            self._sock.sendall(_HDR.pack(MAGIC, VERSION, mtype, rid, 0, len(payload)))
            view = memoryview(payload)
            for o in range(0, len(view), 1 << 30):  # macOS rejects single socket sends of >= 2 GiB (EINVAL)
                self._sock.sendall(view[o:o + (1 << 30)])
            magic, ver, rtype, rrid, flags, n = _HDR.unpack(self._recv(_HDR.size))
            if magic != MAGIC or rrid != rid or rtype != mtype:
                raise RemoteError("protocol desync")
            body = self._recv(n)
        except (OSError, RemoteError) as e:
            # A timeout or error mid-frame leaves the stream at an unknown position: never reuse it (the next
            # call would read this call's late reply).
            self._broken = repr(e)
            raise
        if flags & 1:
            raise RemoteError(body.decode(errors="replace"))
        return body

    # -- API
    def put_blob(self, data, sha: str | None = None) -> str:
        """Stores bytes (bytes or any buffer) on the device under their sha256 (computed unless given) unless already
        there."""
        sha = sha or hashlib.sha256(data).hexdigest()
        if not self._call(HAS_BLOB, sha.encode())[0]:
            self._call(PUT_BLOB, sha.encode() + bytes(data))
        return sha

    def load(self, model, gpu_replay: int | None = None, gpu_segments: int = 0) -> int:
        """Uploads (if missing) and loads a CompiledModel or .lkm file. gpu_replay: None (device default), False/True
        (ICB record/replay off/on); gpu_segments: GPU command buffers in flight on the device (0 = unchanged).
        Servers with the "split_models" feature get the graph and each weight as separate blobs: a weight already
        on the device (same bytes, any model) is never sent again."""
        if isinstance(model, (str, Path)):
            return self._load_path(Path(model), gpu_replay, gpu_segments)
        if self.supports("split_models"):  # split straight from the model: no .lkm written, no copy of the weights
            from . import _lokinfer as _C

            model._drain()
            skeleton, views = _C.lkm_split(model._model, model._externals)
            return self._load_split(memoryview(skeleton), [memoryview(v) for v in views], gpu_replay, gpu_segments)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.lkm"
            model.save(p)
            return self._load_path(p, gpu_replay, gpu_segments)

    def _load_path(self, path: Path, gpu_replay, gpu_segments) -> int:
        if not self.supports("split_models"):
            return self.load_blob(self.put_blob(path.read_bytes()), gpu_replay, gpu_segments)
        import mmap

        from . import _lokinfer as _C

        ranges = _C.lkm_constant_ranges(str(path))
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            data_offset = struct.unpack_from("<Q", mm, 24)[0]  # header: magic, version, meta offset/size, data offset
            view = memoryview(mm)
            try:
                return self._load_split(view[:data_offset], [view[off:off + size] for off, size in ranges],
                                        gpu_replay, gpu_segments)
            finally:
                view.release()

    def _load_split(self, skeleton, constants, gpu_replay, gpu_segments) -> int:
        """LOAD_MODEL_SPLIT from a skeleton and the constants' bytes (buffers): uploads only the missing blobs."""
        from concurrent.futures import ThreadPoolExecutor

        # sha256 releases the GIL: 8 threads hash a 3.4 GB model in ~0.2 s (serial: 1.4 s, M2 Max)
        with ThreadPoolExecutor(8) as ex:
            digests = list(ex.map(lambda b: hashlib.sha256(b).hexdigest(), constants))
        have = self.stored_blobs(digests)  # one round trip instead of one HAS_BLOB per constant
        for b, sha, ok in zip(constants, digests, have):
            if not ok:
                self._call(PUT_BLOB, sha.encode() + bytes(b))
        shas = digests
        skeleton = self.put_blob(skeleton)
        payload = skeleton.encode() + struct.pack("<I", len(shas)) + "".join(shas).encode()
        if gpu_replay is not None or gpu_segments:
            payload += struct.pack("<i", -1 if gpu_replay is None else int(bool(gpu_replay)))
            if gpu_segments:
                payload += struct.pack("<I", gpu_segments)
        return struct.unpack("<I", self._call(LOAD_MODEL_SPLIT, payload))[0]

    def load_blob(self, sha: str, gpu_replay: int | None = None, gpu_segments: int = 0) -> int:
        """Loads a model blob already on the device (put_blob) without uploading anything."""
        payload = sha.encode()
        if gpu_replay is not None or gpu_segments:
            payload += struct.pack("<i", -1 if gpu_replay is None else int(bool(gpu_replay)))
            if gpu_segments:
                payload += struct.pack("<I", gpu_segments)
        return struct.unpack("<I", self._call(LOAD_MODEL, payload))[0]

    def has_blob(self, sha: str) -> bool:
        return bool(self._call(HAS_BLOB, sha.encode())[0])

    def run(self, model_id: int, inputs, warmup: int = 0, iters: int = 1):
        payload = struct.pack("<IIII", model_id, warmup, iters, len(inputs)) + b"".join(_encode_tensor(x) for x in inputs)
        body = self._call(RUN, payload)
        outs, off = self._decode_outputs(body, 0)
        (nt,) = struct.unpack_from("<I", body, off)
        return outs, list(struct.unpack_from(f"<{nt}d", body, off + 4))

    # -- content-addressed test data + RUN2 (feature "run2")
    def put_tensor(self, x) -> Blob:
        """Stores one tensor on the device by its content hash (uploaded only if the device doesn't have it; the
        phone keeps blobs across sessions and reinstalls). Returns the handle for run2 inputs / references."""
        data = _encode_tensor(x)
        sha = hashlib.sha256(data).hexdigest()
        known = self.__dict__.setdefault("_known_blobs", set())
        if sha not in known:
            if not self._call(HAS_BLOB, sha.encode())[0]:
                self._call(PUT_BLOB, sha.encode() + data)
            known.add(sha)
        return Blob(sha)

    def run2(self, model_id: int, inputs, compare=(), return_outputs: bool = True, warmup: int = 0, iters: int = 1):
        """inputs: arrays (sent inline), `Blob`s (put_tensor), `Generated` specs or `DeviceTensor`s (bound in place;
        mutable inputs must be these, and every warmup / timed iteration writes them). compare: (output, ref, rtol,
        atol) with ref a `Blob` or an int (another output of the same call). Returns (outputs or [], [Stats],
        times_ns)."""
        parts = [struct.pack("<IIIII", model_id, warmup, iters, 1 if return_outputs else 0, len(inputs))]
        for x in inputs:
            if isinstance(x, DeviceTensor):
                parts.append(struct.pack("<BI", 3, x.handle))
            elif isinstance(x, Blob):
                parts.append(b"\x01" + x.encode())
            elif isinstance(x, Generated):
                parts.append(b"\x02" + x.encode())
            else:
                parts.append(b"\x00" + _encode_tensor(x))
        parts.append(struct.pack("<I", len(compare)))
        for out, ref, rtol, atol in compare:
            if isinstance(ref, Blob):
                parts.append(struct.pack("<IB", out, 1) + ref.encode() + struct.pack("<dd", rtol, atol))
            else:
                parts.append(struct.pack("<IBIdd", out, 3, int(ref), rtol, atol))
        body = self._call(RUN2, b"".join(parts))
        outs, off = self._decode_outputs(body, 0)
        (nc,) = struct.unpack_from("<I", body, off)
        off += 4
        stats = []
        for _ in range(nc):
            v = struct.unpack_from("<IQQQqdddd", body, off)
            off += struct.calcsize("<IQQQqdddd")
            stats.append(Stats(*v))
        (nt,) = struct.unpack_from("<I", body, off)
        return outs, stats, list(struct.unpack_from(f"<{nt}d", body, off + 4))

    # -- device tensors (feature "device_tensors")
    def alloc_tensor(self, dtype, shape) -> DeviceTensor:
        """A zeroed tensor on the device (server-owned, kept until free_tensor; <= 4 GiB of them in all)."""
        dt = np.dtype(dtype)
        shape = tuple(int(d) for d in shape)
        payload = struct.pack("<II", _DTYPES.index(_dt.from_numpy_dtype(dt)), len(shape)) + struct.pack(f"<{len(shape)}q", *shape)
        (h,) = struct.unpack("<I", self._call(ALLOC_TENSOR, payload))
        return DeviceTensor(h, dt, shape)

    def write_tensor(self, t: DeviceTensor, x) -> None:
        """Copies `x` (same dtype and shape) into the device tensor."""
        a = np.asarray(x)
        if a.dtype != t.dtype or a.shape != t.shape:
            raise ValueError(f"write_tensor: {a.dtype}{list(a.shape)} into {t!r}")
        self._call(WRITE_TENSOR, struct.pack("<I", t.handle) + _encode_tensor(a))

    def read_tensor(self, t: DeviceTensor) -> np.ndarray:
        """The device tensor's current contents."""
        body = self._call(READ_TENSOR, struct.pack("<I", t.handle))
        outs, _ = self._decode_outputs(struct.pack("<I", 1) + body, 0)
        return outs[0]

    def free_tensor(self, t: DeviceTensor) -> None:
        self._call(FREE_TENSOR, struct.pack("<I", t.handle))

    def free_model(self, model_id: int) -> None:
        """Frees a loaded model (and its memory) now instead of at disconnect (feature "free_model")."""
        self._call(FREE_MODEL, struct.pack("<I", model_id))

    def blobs(self) -> dict:
        """The device's blob store: {"count", "bytes"} (feature "blobs")."""
        return json.loads(self._call(BLOBS, struct.pack("<BI", 0, 0)).decode())

    def clear_blobs(self, keep=()) -> dict:
        """Removes every stored blob (weights, models, test tensors) except the sha256 hex digests in `keep`; returns
        {"count", "bytes"} left and {"removed", "removed_bytes"}. Models loaded in this session keep running (their
        mmaps outlive the files); the next upload of a removed blob sends it again."""
        keep = [str(k) for k in keep]
        assert all(len(k) == 64 for k in keep), "keep: sha256 hex digests"
        return json.loads(self._call(BLOBS, struct.pack("<BI", 1, len(keep)) + "".join(keep).encode()).decode())

    def stored_blobs(self, shas) -> list[bool]:
        """Which of these blobs the device has, in one round trip (feature "blobs_query"; else one HAS_BLOB each)."""
        shas = [str(k) for k in shas]
        if not self.supports("blobs_query"):
            return [self.has_blob(s) for s in shas]
        body = self._call(BLOBS, struct.pack("<BI", 3, len(shas)) + "".join(shas).encode()) if shas else b""
        return [b == 1 for b in body]

    def remove_blobs(self, shas) -> dict:
        """Removes the listed blobs (sha256 hex digests; unknown ones are ignored); same result as clear_blobs."""
        shas = [str(k) for k in shas]
        assert all(len(k) == 64 for k in shas), "sha256 hex digests"
        return json.loads(self._call(BLOBS, struct.pack("<BI", 2, len(shas)) + "".join(shas).encode()).decode())

    def memory(self) -> dict:
        """Process memory of the device's executor (bytes): phys_footprint[_peak], resident[_peak], gpu_allocated; and the
        live device tensors (device_tensors, device_tensor_bytes). Feature "memory_pressure" adds: available (iOS: bytes
        left before the app's jetsam limit; -1 on macOS), pressure ("normal" / "warn" / "critical": the system level),
        pressure_events (warn / critical notifications so far), pageins (the process's page-ins so far: diff it around
        a timed region; mapped weights re-read from flash show up here), weights / weights_resident / weights_locked
        (the loaded models' mapped constants), system_free / _file_backed / _compressor / _wired."""
        return json.loads(self._call(MEMORY, b"").decode())

    def pin_model(self, model_id: int, mode: str = "lock") -> dict:
        """Keeps a loaded model's memory-mapped weights in RAM (feature "pin_model"): mode "lock" (mlock: wired, iOS
        cannot evict them, but they count in phys_footprint, against the app's limit), "gpu" (no-copy Metal buffers in
        the GPU residency set: wired by Metal, not in phys_footprint), "willneed" (madvise read-ahead only) or
        "unlock" (drops lock / gpu). Returns {bytes, resident, locked, error} (all or nothing: on failure locked = 0
        and error says why)."""
        m = {"unlock": 0, "willneed": 1, "lock": 2, "gpu": 3}[mode]
        return json.loads(self._call(PIN_MODEL, struct.pack("<II", model_id, m)).decode())

    def memory_check(self) -> tuple[bool, str]:
        """(ok, why): False when the device reports memory pressure (system level warn / critical) — timings taken
        then can include weights re-read from flash (docs/todo.md "fc-thrashing"). Servers without the
        "memory_pressure" feature: always ok."""
        if not self.supports("memory_pressure"):
            return True, "no memory_pressure feature"
        m = self.memory()
        gb = 1 << 30
        why = (f"pressure {m['pressure']}, available {m['available'] / gb:.2f} GB, system free "
               f"{m.get('system_free', 0) / gb:.2f} GB, weights resident {m['weights_resident'] / gb:.2f} / "
               f"{m['weights'] / gb:.2f} GB")
        return m["pressure"] == "normal", why

    @staticmethod
    def _decode_outputs(body: bytes, off: int):
        def take(fmt):
            nonlocal off
            v = struct.unpack_from(fmt, body, off)
            off += struct.calcsize(fmt)
            return v

        (n,) = take("<I")
        outs = []
        for _ in range(n):
            dt_idx, ndim = take("<II")
            shape = take(f"<{ndim}q")
            (nbytes,) = take("<Q")
            dt = _DTYPES[dt_idx]
            outs.append(np.frombuffer(body, dtype=dt.np, count=nbytes // dt.itemsize, offset=off).reshape(shape).copy())
            off += nbytes
        return outs, off

    # -- profiling (docs/profiling.md); needs a server advertising the "profile" / "gpu_inspect" features
    def supports(self, feature: str) -> bool:
        return feature in self.info.get("features", [])

    def set_profile(self, model_id: int, level: int = 1, gpu_serialize: bool = False) -> None:
        """Executable::setProfile on the device: level 1 records every following RUN iteration."""
        self._call(PROFILE, struct.pack("<III", model_id, level, 1 if gpu_serialize else 0))

    def take_profile(self, model_id: int) -> dict:
        """The calls recorded since the last take (Executable::takeProfileJson), parsed."""
        return json.loads(self._call(TAKE_PROFILE, struct.pack("<I", model_id)).decode())

    def gpu_inspect(self, functions=(), archive: bool = False) -> tuple[dict, bytes]:
        """Device info + pipeline statistics of the device's metallib functions; with archive=True also the
        serialized binary archive (native GPU code) of those functions."""
        body = self._call(GPU_INSPECT, bytes([1 if archive else 0]) + "\n".join(functions).encode())
        (n,) = struct.unpack_from("<Q", body, 0)
        return json.loads(body[8:8 + n].decode()), body[8 + n:]

    def shutdown(self) -> None:
        self._call(SHUTDOWN, b"")
