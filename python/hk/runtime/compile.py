"""hipcc, with a content-addressed cache, a spill gate and a wait gate.

A hipcc invocation for a real kernel costs 20-40 s, which is too slow to pay on
every import and far too slow to pay inside an autotune sweep. So the compiled
.so is keyed by everything that can change its bytes:

  * the generated source (which already folds in the kernel's Python source,
    its constexpr arguments and the target, since all three shaped the text),
  * the toolchain -- hipcc's own version string and the flags,
  * the contents of the include directories (hk's and HipKittens'), because
    editing a header changes the kernel while leaving the generator's output
    byte-identical. This is the one people get wrong, and it is the one that
    produces a stale kernel that looks right.

Cache entries are never invalidated, only added: a key that hashes everything
does not need eviction for correctness, and `HK_CACHE_DIR` is a directory the
user can delete.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..target import Target, get_target
from . import resources
from .resources import KernelResources, ResourceError


class CompileError(Exception):
    pass


# ---------------------------------------------------------------- paths

def include_dirs() -> List[Path]:
    """Where the generated C++ finds its headers: hk's own (pyutils/hk_bind.cuh)
    and HipKittens' (kittens.cuh and the arch libraries), in that order.

    An installed wheel carries both, merged, in hk/include. A checkout has hk's
    in include/ and HipKittens' in the third_party/HipKittens submodule, pinned
    to the commit the passes were validated against -- they encode facts about
    those headers, down to how many ds_reads a load issues. HK_INCLUDE_DIR
    overrides, as a path list."""
    env = os.environ.get("HK_INCLUDE_DIR")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    here = Path(__file__).resolve()
    packaged = here.parents[1] / "include"
    if (packaged / "kittens.cuh").exists():
        return [packaged]
    repo = here.parents[3]
    kittens = repo / "third_party" / "HipKittens" / "include"
    if (kittens / "kittens.cuh").exists():
        return [repo / "include", kittens]
    raise CompileError(
        f"cannot find the HipKittens headers: no {packaged / 'kittens.cuh'} and "
        f"no {kittens / 'kittens.cuh'}. In a checkout, run `git submodule "
        f"update --init`; or set HK_INCLUDE_DIR to the directories to search."
    )


def cache_dir() -> Path:
    d = Path(
        os.environ.get("HK_CACHE_DIR")
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hk"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def hipcc() -> str:
    for cand in (
        os.environ.get("HK_HIPCC"),
        shutil.which("hipcc"),
        "/opt/rocm/bin/hipcc",
    ):
        if cand and os.path.exists(cand):
            return cand
    raise CompileError(
        "no hipcc found. Set HK_HIPCC, or put ROCm's bin on PATH. hk compiles "
        "with the stock hipcc on purpose -- it needs no custom toolchain, which "
        "is what lets a generated kernel run in any ROCm container."
    )


# ---------------------------------------------------------------- flags


def _pybind_includes() -> List[str]:
    flags = [f"-I{sysconfig.get_path('include')}"]
    try:
        import pybind11  # noqa: PLC0415

        flags.append(f"-I{pybind11.get_include()}")
    except ImportError:
        # torch vendors the same headers; on a pod that has torch but not a
        # standalone pybind11 this is the path that works.
        try:
            import torch  # noqa: PLC0415

            flags.append(f"-I{Path(torch.__file__).parent / 'include'}")
        except ImportError:
            raise CompileError(
                "neither pybind11 nor torch is importable, so there are no "
                "pybind11 headers to compile against. pip install pybind11."
            ) from None
    return flags


def ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def rocm_path() -> Path:
    env = os.environ.get("ROCM_PATH")
    if env:
        return Path(env)
    # hipcc lives at <rocm>/bin/hipcc or <rocm>/lib/llvm/bin; walk up to the
    # directory that has include/hip.
    for parent in Path(hipcc()).resolve().parents:
        if (parent / "include" / "hip").is_dir():
            return parent
    return Path("/opt/rocm")


def flags(target: Target, module_name: str, extra: Sequence[str] = ()) -> List[str]:
    """The hipcc command line, mirroring kernels/common.mk BUILD_MODE=pyext.

    include/hip is on the path explicitly because base_types.cuh includes
    <hip_bf16.h> unqualified; hipcc's own default search does not cover it.
    """
    return [
        f"-D{target.kittens_define}",
        f"--offload-arch={target.arch}",
        "-std=c++20",
        "-O3",
        "-w",
        *(f"-I{d}" for d in include_dirs()),
        f"-I{rocm_path() / 'include' / 'hip'}",
        *_pybind_includes(),
        f"-DHK_MODULE_NAME={module_name}",
        "-shared",
        "-fPIC",
        # The gate depends on this flag; see runtime/resources.
        "-Rpass-analysis=kernel-resource-usage",
        *extra,
    ]


def codeobj_flags(target: Target, extra: Sequence[str] = ()) -> List[str]:
    """The hipcc command line for a device-only code object: no host pass, so
    no pybind or libtorch headers to parse -- which is most of a build
    (tools/hk-bench/compile_split: 4.4 s for the pybind module against 0.6-0.9
    for the code object, here; 17 s for the torch op on the pod). A raw ELF,
    not an offload bundle, so its symbols can be read without a GPU."""
    return [
        f"-D{target.kittens_define}",
        f"--offload-arch={target.arch}",
        "-std=c++20",
        "-O3",
        "-w",
        *(f"-I{d}" for d in include_dirs()),
        f"-I{rocm_path() / 'include' / 'hip'}",
        "--offload-device-only",
        "--genco",
        "--no-gpu-bundle-output",
        "-Rpass-analysis=kernel-resource-usage",
        *extra,
    ]


# ---------------------------------------------------------------- key


def _tree_hash(root: Path, suffixes=(".cuh", ".hpp", ".h")) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in suffixes:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


_toolchain_cache: dict = {}


def _hipcc_version() -> str:
    exe = hipcc()
    key = ("version", exe)
    if key not in _toolchain_cache:
        try:
            _toolchain_cache[key] = subprocess.run(
                [exe, "--version"], capture_output=True, text=True, timeout=120
            ).stdout
        except Exception as e:
            raise CompileError(f"{exe} --version failed: {e}") from e
    return _toolchain_cache[key]


def hipcc_fingerprint() -> str:
    """hipcc's version alone: unlike toolchain_fingerprint, it does not move
    when a header no kernel of this arch includes does."""
    return hashlib.sha256(_hipcc_version().encode()).hexdigest()[:16]


def toolchain_fingerprint() -> str:
    """hipcc version + the include tree. Cached per process; hashing include/
    is ~10 ms and the answer cannot change mid-run."""
    exe = hipcc()
    if exe in _toolchain_cache:
        return _toolchain_cache[exe]
    fp = hashlib.sha256(
        (_hipcc_version() + "".join(_tree_hash(d) for d in include_dirs())).encode()
    ).hexdigest()[:32]
    _toolchain_cache[exe] = fp
    return fp


#: The HipKittens tree each -DKITTENS_* define selects (include/kittens.cuh).
_ARCH_TREE = {"KITTENS_CDNA3": "cdna3", "KITTENS_CDNA4": "cdna4", "KITTENS_CDNA5": "cdna5",
              "KITTENS_RDNA3": "rdna3", "KITTENS_RDNA4": "rdna4"}


def target_headers_fingerprint(target: Target) -> str:
    """hipcc's version + only the headers a kernel of `target` can include: kittens.cuh,
    pyutils/, the one arch tree its -DKITTENS_* define selects, and its extra_includes
    (hk/cdna.cuh for CDNA). A change to the CDNA shim then recompiles CDNA kernels and leaves
    every RDNA one cached, and the other way round -- toolchain_fingerprint() moved every
    kernel's key on any header, so a branch touching include/hk/cdna.cuh compiled the whole
    codegen tier cold (40-56 min on CI against main's 5). A HipKittens tree of a layout this
    does not know is hashed whole. Cached per process and target."""
    key = ("target-headers", hipcc(), target.arch)
    if key in _toolchain_cache:
        return _toolchain_cache[key]
    tree = _ARCH_TREE.get(getattr(target, "kittens_define", ""), "")
    extra = [e.strip('"<>') for e in getattr(target, "extra_includes", ())]
    h = hashlib.sha256(_hipcc_version().encode())
    for d in include_dirs():     # what generated C++ includes: kittens.cuh + extra_includes
        if (d / "kittens.cuh").exists():
            if tree and (d / tree).is_dir():
                h.update((d / "kittens.cuh").read_bytes())
                for sub in ("pyutils", tree):
                    if (d / sub).is_dir():
                        h.update(_tree_hash(d / sub).encode())
            else:                  # a layout this does not know: all of it
                h.update(_tree_hash(d).encode())
        for e in extra:
            if (d / e).is_file():
                h.update(e.encode())
                h.update((d / e).read_bytes())
    fp = h.hexdigest()[:32]
    _toolchain_cache[key] = fp
    return fp


#: Stand-in for the module name while keying. The real name is derived from the
#: key, so it cannot be an input to it; it also cannot change the generated code
#: in any way that matters, since it only names the pybind entry point.
_KEY_MODULE = "<module>"


_dialect_version: list = []


def dialect_version() -> str:
    """The hk dialect's semantic version, or "none" without hk._mlir. Not a
    hash of the library: a rebuild that changes nothing it means must not
    throw the compile cache away."""
    if not _dialect_version:
        try:
            from .._mlir.dialects import hk as hkd  # noqa: PLC0415

            _dialect_version.append(hkd.dialect_version())
        except Exception:  # noqa: BLE001
            _dialect_version.append("none")
    return _dialect_version[0]


def cache_key(source: str, target: Target, extra: Sequence[str] = (),
              kind: str = "so") -> str:
    """Everything that can change the compiled bytes.

    The full flag list is in here, not just `extra`: hk's own flags change when
    hk changes, and a cache that ignores them serves the artifact from before
    the fix. That is exactly the failure mode the include-path bug would have
    hidden.
    """
    h = hashlib.sha256()
    h.update(target_headers_fingerprint(target).encode())
    h.update(dialect_version().encode())
    fl = (flags(target, _KEY_MODULE, extra) if kind == "so"
          else codeobj_flags(target, extra))
    h.update("\0".join(fl).encode())
    h.update(source.encode())
    return h.hexdigest()[:32]


#: What a device-only build leaves in its cache directory.
CODEOBJ = "kernel.co"

#: sha256 of every generated source this process has built or loaded, with
#: its flags but not its include paths.
_BUILT: set = set()


def built_fingerprint() -> str:
    """One hash over the generated C++ (and hipcc flags) of every kernel this
    process has built or taken from the cache, or "" if none. Two processes
    that ran the same kernels agree on it whatever hk's version, the
    dialect's or the checkout's path; a change to any of those kernels' code
    changes it. What the C++ includes is toolchain_fingerprint()'s."""
    if not _BUILT:
        return ""
    return hashlib.sha256("".join(sorted(_BUILT)).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- build


@dataclass
class Build:
    """One compiled artifact and everything used to judge it."""

    key: str
    module_name: str
    so_path: Path
    source_path: Path
    log_path: Path
    kernels: List[KernelResources]
    cached: bool

    def report(self) -> str:
        """One line per instantiation: vgpr, occupancy, LDS, scratch, spills
        (KernelResources.line). `python -m hk.check` reports the same with
        the occupancy LDS allows."""
        return resources.report(self.kernels)


_WAIT_MARK = "WAITS"


def _note_unused_frames(kernels, binary: Path, kind: str, log: Optional[Path] = None) -> None:
    """A kernel with ScratchSize > 0 but no VGPR spill: if its code object shows the frame
    reserved and never accessed (cocheck.frame_verdict -- no scratch_*, no flat_* memory, no
    flat scratch, no private segment buffer SGPRs), record why on it, so the spill gate lets it
    through, and say so in the build log. Anything else stays a refusal."""
    todo = [k for k in kernels if k.scratch and not k.vgpr_spill and not k.scratch_note]
    if not todo:
        return
    from . import cocheck  # noqa: PLC0415

    try:
        verdict = cocheck.unused_scratch_frame(binary, bundled=(kind == "so"))
    except Exception:  # noqa: BLE001 -- no verdict, no pass: the gate refuses as before
        return
    for k in todo:
        ok, _why = verdict.get(k.mangled, verdict.get(k.name, (False, "")))
        if ok:
            k.scratch_note = cocheck.UNUSED_FRAME
            line = (f"hk: {k.name}: ScratchSize={k.scratch} B/lane allowed -- {cocheck.UNUSED_FRAME}"
                    f" (no scratch_* / flat_* access, no private segment buffer or flat scratch)")
            print(line, file=sys.stderr)
            if log is not None:
                try:
                    with open(log, "a") as f:
                        f.write(line + "\n")
                except OSError:
                    pass


def _wait_gate(binary: Path, kind: str) -> bool:
    """Refuse a binary whose code reads registers an LDS load is still filling (cocheck's
    lds_copy_hazards): the compiler copied an inline-asm ds_read's destination before hk's
    wait, and the kernel is wrong though nothing spilled. HK_WAIT_GATE=0 turns it off, for
    looking at such a kernel, never for running one. Returns whether it checked (a cache
    entry is marked checked only then)."""
    if os.environ.get("HK_WAIT_GATE", "1") in ("0", "off", "no"):
        return False
    from . import cocheck  # noqa: PLC0415

    found = cocheck.lds_copy_hazards(binary, bundled=(kind == "so"))
    if found:
        shown = "\n".join(f"  {h}" for h in found[:6])
        more = f"\n  ... {len(found) - 6} more" if len(found) > 6 else ""
        raise ResourceError(
            f"generated kernel rejected: {len(found)} read(s) of registers an LDS load is "
            f"still filling, before the wait that retires it:\n{shown}{more}\n"
            "The compiler copied the destination of hk's inline-asm LDS read (to build an "
            "operand tuple, or into an AGPR under register pressure) ahead of hk's lgkmcnt "
            "wait, so the copy holds stale data. A kernel like this is not slow, it is wrong. "
            "Lower the register pressure: a smaller tile, fewer values live across the "
            "K loop, fewer stages.")
    return True


def _load_cached(d: Path, target: Target) -> Optional[Build]:
    so = (next(d.glob("*.so"), None) or next(d.glob("*.*.so"), None)
          or ((d / CODEOBJ) if (d / CODEOBJ).exists() else None))
    marker = d / "OK"
    if so is None or not marker.exists():
        return None
    log = d / "build.log"
    try:
        ks = resources.parse(log.read_text(), target) if log.exists() else []
    except ResourceError:
        # A cache entry whose log we cannot read is still a valid .so -- the
        # gate passed when it was written, which is what the OK marker records.
        ks = []
    return Build(
        key=d.name,
        module_name=marker.read_text().strip(),
        so_path=so,
        source_path=d / "kernel.cpp",
        log_path=log,
        kernels=ks,
        cached=True,
    )


#: Armed by an engine plugin that applied routes (`arm_jit_notice`): the first kernel this
#: process has to compile then says so, once, with the command that would have built it ahead
#: of time. A JIT compile on the serving path is seconds a kernel in every worker; a request
#: that waits for one looks like a hang. [hint] while armed, [] otherwise.
_JIT_NOTICE: List[str] = []
_JIT_SAID: List[bool] = []

#: What the notice tells a server to run first.
WARM_HINT = "python3 -m hk.integration warm-routes (and --warm for attention)"


def arm_jit_notice(what: str) -> None:
    """Make this process's first JIT compile print one line saying it is compiling, and how
    to pre-build (WARM_HINT). `what` names who armed it ("vLLM plugin")."""
    if not _JIT_NOTICE:
        _JIT_NOTICE.append(what)


def _jit_notice(name: str) -> None:
    if _JIT_NOTICE and not _JIT_SAID:
        _JIT_SAID.append(True)
        print(f"[hk] JIT: compiling kernel {name!r} on first use (pid {os.getpid()}): the "
              f"{_JIT_NOTICE[0]}'s routes were not pre-warmed into {cache_dir()}, so this call "
              f"waits for hipcc (seconds a kernel, in every worker). Run {WARM_HINT} with the "
              f"server's HK_CACHE_DIR before starting the server.", file=sys.stderr, flush=True)


def build(
    source: str,
    arch: str,
    *,
    name: str = "hk_kernel",
    extra_flags: Sequence[str] = (),
    allow_scratch: bool = False,
    max_vgprs: Optional[int] = None,
    min_occupancy: Optional[int] = None,
    verbose: bool = False,
    kind: str = "so",
) -> Build:
    """Compile generated C++ into a loadable extension, or return the cached one.

    `kind` "so" is an importable extension (pybind or torch registration),
    "co" a device-only code object for hk.runtime.codeobj -- same cache, same
    gate. Raises ResourceError, and leaves nothing importable behind, if any
    instantiation spills, or if its code reads registers an LDS load is still
    filling (the wait gate, `_wait_gate`; a cache entry from before the gate is
    checked on its first hit).
    """
    if kind not in ("so", "co"):
        raise ValueError(f"build kind {kind!r}; 'so' or 'co'")
    target = get_target(arch)
    key = cache_key(source, target, extra_flags, kind)
    out = cache_dir() / key
    fl = (flags(target, _KEY_MODULE, extra_flags) if kind == "so"
          else codeobj_flags(target, extra_flags))
    _BUILT.add(hashlib.sha256("\0".join(
        [source, *(f for f in fl if not f.startswith("-I"))]).encode()).hexdigest())

    if (hit := _load_cached(out, target)) is not None:
        # Re-gate on a hit. The key covers the source and the flags but not the
        # thresholds, so the same bytes can be asked for by one kernel with no
        # occupancy requirement and by another with a strict one; whoever
        # compiled first must not decide for the second. Cheap -- the numbers
        # come from the cached log, not from hipcc. Entries whose log no longer
        # parses gate on nothing, which is what the OK marker already attests.
        if hit.kernels:
            _note_unused_frames(hit.kernels, hit.so_path, kind)
            resources.gate(
                hit.kernels,
                allow_scratch=allow_scratch,
                max_vgprs=max_vgprs,
                min_occupancy=min_occupancy,
            )
        if not (out / _WAIT_MARK).exists() and _wait_gate(hit.so_path, kind):
            # built before the wait gate, or with it off: checked once, then marked
            try:
                (out / _WAIT_MARK).write_text("lds copies checked\n")
            except OSError:
                pass
        return hit

    module_name = f"hk_{name}_{key[:12]}"
    so_name = module_name + ext_suffix() if kind == "so" else CODEOBJ
    _jit_notice(name)

    # Build in a scratch directory and move into place, so a killed or failing
    # compile never leaves a half-written .so that the next run would import.
    tmp = Path(tempfile.mkdtemp(prefix="hk-build-", dir=cache_dir()))
    moved = False
    try:
        src = tmp / "kernel.cpp"
        src.write_text(source)
        fl = (flags(target, module_name, extra_flags) if kind == "so"
              else codeobj_flags(target, extra_flags))
        cmd = [hipcc(), *fl, str(src), "-o", str(tmp / so_name)]
        if verbose:
            print(" ".join(cmd), file=sys.stderr)

        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = proc.stdout + proc.stderr
        (tmp / "build.log").write_text(log)
        (tmp / "command").write_text(" ".join(cmd))

        if proc.returncode != 0:
            raise CompileError(
                f"hipcc failed ({proc.returncode}) for kernel {name!r}.\n"
                f"Source kept at {_keep(tmp, key)}\n\n{log}"
            )

        kernels = resources.parse(log, target)
        _note_unused_frames(kernels, tmp / so_name, kind, tmp / "build.log")
        try:
            resources.gate(
                kernels,
                allow_scratch=allow_scratch,
                max_vgprs=max_vgprs,
                min_occupancy=min_occupancy,
            )
            checked = _wait_gate(tmp / so_name, kind)
        except ResourceError as e:
            raise ResourceError(
                f"{e}\n\nGenerated source kept at {_keep(tmp, key)}"
            ) from None

        if checked:
            (tmp / _WAIT_MARK).write_text("lds copies checked\n")
        (tmp / "OK").write_text(module_name)
        try:
            os.rename(tmp, out)
            moved = True
        except OSError:
            # A concurrent builder won the race. Its artifact is byte-identical
            # by construction -- the key hashes everything that could differ --
            # so losing is not an error.
            if not (out / "OK").exists():
                raise
            return _load_cached(out, target)
    finally:
        if not moved:
            shutil.rmtree(tmp, ignore_errors=True)

    return Build(
        key=key,
        module_name=module_name,
        so_path=out / so_name,
        source_path=out / "kernel.cpp",
        log_path=out / "build.log",
        kernels=kernels,
        cached=False,
    )


def _keep(tmp: Path, key: str) -> Path:
    """Preserve a failed build for inspection. A failure whose source is gone
    is much harder to act on than one that costs a few KB of disk."""
    dest = cache_dir() / f"failed-{key}"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(tmp, dest)
    return dest
