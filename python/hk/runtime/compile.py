"""hipcc, with a content-addressed cache, a spill gate and a wait gate.

A hipcc invocation for a real kernel costs 20-40 s, which is too slow to pay on
every import and far too slow to pay inside an autotune sweep. So the compiled
.so is keyed by everything that can change its bytes:

  * the generated source (which already folds in the kernel's Python source,
    its constexpr arguments and the target, since all three shaped the text),
  * the toolchain -- hipcc's own version string and the flags (and torch's and
    pybind11's versions, when their headers are on the include path),
  * the contents of every header the kernel's compile reads, from hk's and
    HipKittens' include directories or any other, listed by the compiler's own
    preprocessor (`_scan_deps`), because editing a header changes the kernel
    while leaving the generator's output byte-identical. This is the one people
    get wrong, and it is the one that produces a stale kernel that looks right.

Cache entries are never invalidated, only added: a key that hashes everything
does not need eviction for correctness, and `HK_CACHE_DIR` is a directory the
user can delete.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
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
    # directory that has include/hip. TheRock's pip layout (rocm/pytorch 10.x) puts
    # a launcher script in /usr/local/bin instead, so the clang hipcc runs is asked
    # too: its InstalledDir is <rocm>/lib/llvm/bin, under _rocm_sdk_devel.
    for start in (lambda: Path(hipcc()).resolve(), clang_dir):
        where = start()
        for parent in (where.parents if where is not None else ()):
            if (parent / "include" / "hip").is_dir():
                return parent
    return Path("/opt/rocm")


def clang_dir() -> Optional[Path]:
    """The directory of the clang hipcc runs: HIP_CLANG_PATH when set (hipcc reads
    it), else the `InstalledDir:` of `hipcc --version`. None if neither names one."""
    env = os.environ.get("HIP_CLANG_PATH")
    if env and os.path.isdir(env):
        return Path(env)
    try:
        version = _hipcc_version()
    except CompileError:
        return None
    m = re.search(r"^InstalledDir:\s*(.+?)\s*$", version, re.M)
    return Path(m.group(1)) if m and os.path.isdir(m.group(1)) else None


def compiler_version() -> str:
    """The `clang version` line of `hipcc --version` (its first line when it has none)."""
    try:
        lines = _hipcc_version().splitlines()
    except CompileError as e:
        return str(e)
    return next((l.strip() for l in lines if "clang version" in l),
                lines[0].strip() if lines else "unknown")


def llvm_tool(name: str, required: bool = True) -> Optional[str]:
    """ROCm's LLVM tool `name` (llvm-objdump, llvm-readelf, clang-offload-bundler, ...),
    found where hipcc's clang is rather than on PATH: beside that clang (`clang_dir`),
    then <rocm>/llvm/bin and <rocm>/lib/llvm/bin, then PATH. TheRock's pip layout puts
    none of them on PATH. Raises FileNotFoundError naming where it looked when
    `required`, else returns None."""
    dirs: List[Path] = []
    cd = clang_dir()
    if cd is not None:
        dirs.append(cd)
    root = rocm_path()
    dirs += [root / "llvm" / "bin", root / "lib" / "llvm" / "bin"]
    seen = set()
    for d in dirs:
        if d in seen:
            continue
        seen.add(d)
        p = d / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    found = shutil.which(name)
    if found or not required:
        return found
    raise FileNotFoundError(
        f"{name} not found: hk needs ROCm's LLVM tools to read the code objects it builds, "
        f"and looked for it in {', '.join(str(d) for d in dict.fromkeys(dirs))} and on PATH. "
        f"Set ROCM_PATH to the ROCm install that has llvm/bin/{name} (or HIP_CLANG_PATH to "
        f"the directory of the clang hipcc runs, if {name} is beside it).")


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


#: Stand-in for the module name while keying. The real name is derived from the
#: key, so it cannot be an input to it; it also cannot change the generated code
#: in any way that matters, since it only names the pybind entry point.
_KEY_MODULE = "<module>"

#: Names what the key covers. A cache entry written under another scheme is not
#: wrong to keep, only never found again.
_KEY_SCHEME = "hk-key-2: per-kernel header deps"


# ---------------------------------------------------------------- headers
#
# A kernel's key covers the contents of every header its compile reads -- not
# a guess at which trees it might include. The list comes from the compiler
# (a preprocessor pass with clang's -dependency-file, `_scan_deps`), is kept
# beside the cache as a manifest named by the header-free part of the key, and
# on a lookup the listed files are hashed again (each once a process, re-read
# only if its stat changes). An edit to any header the kernel includes, however
# deep, moves its key; an edit to one it does not include leaves it cached.
#
# The previous key hashed kittens.cuh, pyutils/, the arch tree and the target's
# extra_includes (hk/cdna.cuh) -- but only those files: include/hk/cdna_agpr.cuh,
# which cdna.cuh includes, and include/pyutils/hk_bind.cuh, which every pybind
# module includes, were in no key, and an edit to either served the old kernel.

#: path -> ((st_ino, st_size, st_mtime_ns), sha256 of the contents, when it was hashed)
_FILE_HASH: dict = {}

#: A file modified this close before it was hashed is hashed again, as git treats a
#: "racily clean" entry: the file system's clock ticks in milliseconds, so an edit
#: of the same size in the same tick leaves the stat unchanged.
_RACY_NS = 2_000_000_000


def _file_hash(path: str) -> str:
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    sig = (st.st_ino, st.st_size, st.st_mtime_ns)
    got = _FILE_HASH.get(path)
    if got is not None and got[0] == sig and st.st_mtime_ns < got[2] - _RACY_NS:
        return got[1]
    now = time.time_ns()
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "missing"
    _FILE_HASH[path] = (sig, digest, now)
    return digest


def headers_digest(deps: Sequence[str]) -> str:
    """One hash over the paths and contents of `deps` (a missing file hashes as
    missing, so its key misses and the next build rescans)."""
    h = hashlib.sha256()
    for p in deps:
        h.update(f"{p}\0{_file_hash(p)}\n".encode())
    return h.hexdigest()


def _full_key(base: str, digest: str) -> str:
    return hashlib.sha256(f"{base}\0{digest}".encode()).hexdigest()[:32]


def _manifest(base: str) -> Path:
    return cache_dir() / "deps" / base


def _read_manifest(base: str) -> Optional[List[str]]:
    try:
        lines = _manifest(base).read_text().split("\n")
    except OSError:
        return None
    if not lines or lines[0] != _KEY_SCHEME:
        return None
    return [p for p in lines[1:] if p]


def _write_manifest(base: str, deps: Sequence[str]) -> None:
    m = _manifest(base)
    try:
        m.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".deps-", dir=m.parent)
        with os.fdopen(fd, "w") as f:
            f.write("\n".join([_KEY_SCHEME, *deps]) + "\n")
        os.replace(tmp, m)
    except OSError:
        pass          # no manifest: the next lookup rescans, which costs a preprocessor pass


def _parse_depfile(text: str) -> List[str]:
    """The prerequisites of a make rule as clang writes it (`target: a b \\` ...;
    a space in a path is `\\ `, a `$` is `$$`)."""
    text = text.replace("\\\n", " ")
    _, _, rest = text.partition(":")
    toks = re.findall(r"(?:\\.|[^\s\\])+", rest)
    return [re.sub(r"\\(.)", r"\1", t).replace("$$", "$") for t in toks]


def _versioned_dirs(fl: Sequence[str]) -> List[str]:
    """Include directories keyed by a version rather than by their headers' contents:
    ROCm's (hipcc's version), Python's (its path names the version), and pybind11's and
    torch's (`_external_versions`). Hashing torch's few thousand headers on every start
    would buy nothing the version does not."""
    dirs = [str(rocm_path()), sysconfig.get_path("include"), sysconfig.get_path("platinclude")]
    for f in fl:
        if f.startswith("-I") and (_is_torch_include(f[2:]) or _is_pybind_include(f[2:])):
            dirs.append(f[2:])
    return [os.path.realpath(d) for d in dirs if d]


def _under(p: str, d: str) -> bool:
    return p == d or p.startswith(d + os.sep)


#: (directory, package file) -> whether the directory is inside that package; real paths
#: cost syscalls, and a warm lookup asks for every -I on every kernel.
_INSIDE: dict = {}


def _inside_package(d: str, mod: str, sub: str = "") -> bool:
    m = sys.modules.get(mod)
    f = getattr(m, "__file__", None)
    if not f:
        return False
    k = (d, f, sub)
    if k not in _INSIDE:
        root = os.path.realpath(os.path.join(os.path.dirname(f), sub))
        _INSIDE[k] = _under(os.path.realpath(d), root)
    return _INSIDE[k]


def _is_torch_include(d: str) -> bool:
    """`d` is inside the imported torch package (torch_flags, or pybind11 as torch vendors it)."""
    return _inside_package(d, "torch")


def _is_pybind_include(d: str) -> bool:
    """`d` is inside the imported pybind11 package (its headers ship in it)."""
    return _inside_package(d, "pybind11")


def _external_versions(fl: Sequence[str]) -> List[str]:
    """The versions that stand for the headers `_versioned_dirs` leaves out."""
    out = []
    incs = [f[2:] for f in fl if f.startswith("-I")]
    t = sys.modules.get("torch")
    if t is not None and any(_is_torch_include(d) for d in incs):
        out.append(f"torch {t.__version__} {getattr(getattr(t, 'version', None), 'git_version', '')}")
    p = sys.modules.get("pybind11")
    if p is not None and any(_is_pybind_include(d) for d in incs):
        out.append(f"pybind11 {getattr(p, '__version__', '?')}")
    return out


def _scan_deps(fl: Sequence[str], src: Path, kind: str, tmp: Path) -> List[str]:
    """Every header the compile of `src` with `fl` reads, but the toolchain's and
    `_versioned_dirs`', as real paths, sorted. One preprocessor pass per side the
    compile builds -- the device's, and for a .so the host's too: hipcc's -MD writes
    the host side's list only, and with --offload-device-only none at all, so a header
    included under __HIP_DEVICE_COMPILE__ would be missed. -dependency-file, like
    -MMD, leaves system headers out. About 0.1 s a side, against seconds of compile.
    Raises CompileError with the preprocessor's output if a pass fails."""
    drop = {"--genco", "--no-gpu-bundle-output", "-shared", "-fPIC",
            "--offload-device-only", "--offload-host-only"}
    base = [("-DHK_MODULE_NAME=hk_deps" if f.startswith("-DHK_MODULE_NAME=") else f)  # a token
            for f in fl if f not in drop and not f.startswith(("-Rpass", "-L", "-l", "-Wl,"))]
    sides = ["--offload-device-only"] + (["--offload-host-only"] if kind == "so" else [])
    skip = _versioned_dirs(fl)
    found = set()
    for i, side in enumerate(sides):
        dep = tmp / f"deps{i}.d"
        cmd = [hipcc(), *base, side, "-E", str(src), "-o", os.devnull,
               "-Xclang", "-dependency-file", "-Xclang", str(dep),
               "-Xclang", "-MT", "-Xclang", "hkdeps"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not dep.exists():
            raise CompileError(f"hipcc's preprocessor failed ({proc.returncode}) listing the "
                               f"headers of {src}:\n{proc.stdout}{proc.stderr}")
        for p in _parse_depfile(dep.read_text()):
            found.add(os.path.realpath(p))
    found.discard(os.path.realpath(src))
    return sorted(p for p in found if not any(_under(p, d) for d in skip))


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


#: Environment hipcc itself reads that changes what it compiles: flags it appends to every
#: command, and which clang it runs (whose version `hipcc --version` may not report).
_HIPCC_ENV = ("HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND", "HIP_CLANG_PATH")


def cache_key(source: str, target: Target, extra: Sequence[str] = (),
              kind: str = "so") -> str:
    """Everything that can change the compiled bytes but the headers' contents:
    hipcc's version, the dialect's, the full flag list (and the flags hipcc
    appends from its environment), torch's and pybind11's versions when their
    headers are on it, and the source. `build` adds the
    headers (`headers_digest` of the kernel's own dependency list) to make the
    key of a cache entry; this part names its dependency manifest.

    The full flag list is in here, not just `extra`: hk's own flags change when
    hk changes, and a cache that ignores them serves the artifact from before
    the fix. That is exactly the failure mode the include-path bug would have
    hidden.
    """
    h = hashlib.sha256(_KEY_SCHEME.encode())
    h.update(_hipcc_version().encode())
    h.update(dialect_version().encode())
    fl = (flags(target, _KEY_MODULE, extra) if kind == "so"
          else codeobj_flags(target, extra))
    h.update("\0".join(fl).encode())
    h.update("\0".join(_external_versions(fl)).encode())
    h.update("\0".join(f"{v}={os.environ.get(v, '')}" for v in _HIPCC_ENV).encode())
    h.update(kind.encode())
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


_MFMA_MARK = "MFMAHAZ"


def _mfma_mode() -> str:
    """HK_MFMA_GATE: fix (the default: pad and rebuild), refuse, report (say so on stderr
    and in the build log, and build: for a sweep, never for running a kernel), or off."""
    v = os.environ.get("HK_MFMA_GATE", "fix").strip().lower()
    if v in ("0", "off", "no"):
        return "off"
    return v if v in ("refuse", "report") else "fix"


def _mfma_found(binary: Path, kind: str, arch: str) -> Optional[List[str]]:
    """The MFMA hazards of a binary (cocheck.mfma_hazards), [] when its target
    has no MFMAs or its compiler pads every path (mfmahaz.compiler_fixed:
    llvm/llvm-project#225989), None when the gate is off."""
    if _mfma_mode() == "off":
        return None
    if not (arch.startswith("gfx94") or arch == "gfx950"):
        return []
    from . import cocheck, mfmahaz  # noqa: PLC0415

    if mfmahaz.compiler_fixed(cocheck._tool("llc")):
        return []
    return cocheck.mfma_hazards(binary, bundled=(kind == "so"))


def _mfma_refusal(found: Sequence[str]) -> ResourceError:
    shown = "\n".join(f"  {h}" for h in found[:6])
    more = f"\n  ... {len(found) - 6} more" if len(found) > 6 else ""
    return ResourceError(
        f"generated kernel rejected: {len(found)} MFMA result(s) read or overwritten too few "
        f"wait states after the MFMA on some path into the reader:\n{shown}{more}\n"
        "This compiler pads the shortest path into a block too little when it finds a longer "
        "one first (fixed in LLVM by llvm/llvm-project#225989; ROCm/llvm-project#5153), and "
        "the hardware does not interlock on MFMA results: a kernel like this can compute wrong "
        "values. HK_MFMA_GATE=refuse asked for this refusal; the default (fix) rebuilds the "
        "kernel with the s_nop padding the fixed compiler emits.")


def _device_step(argv: Sequence[str]) -> bool:
    """A device compile step of hipcc's pipeline: its -triple is amdgcn's (the host steps
    name amdgcn too, as -aux-triple)."""
    return "-triple" in argv and argv[argv.index("-triple") + 1].startswith("amdgcn")


def _device_asm_step(argv: Sequence[str]) -> bool:
    return "-S" in argv and "-o" in argv and _device_step(argv)


def _plan(cmd: Sequence[str], tmp: Path) -> List[List[str]]:
    """`hipcc -### -save-temps` of `cmd`: its steps, each an argv."""
    import shlex  # noqa: PLC0415

    plan = subprocess.run([*cmd, "-###", "-save-temps"], cwd=tmp, capture_output=True,
                          text=True)
    steps = [shlex.split(l) for l in (plan.stdout + plan.stderr).splitlines()
             if l.startswith(' "')]
    if plan.returncode != 0 or not steps:
        raise CompileError(f"hipcc -### failed for the MFMA padding rebuild:\n{plan.stderr}")
    return steps


def _pad_mfma_hazards(cmd: Sequence[str], tmp: Path, arch: str) -> List[str]:
    """Rebuild `cmd` (hipcc, output into `tmp`) through its own pipeline, as
    `hipcc -### -save-temps` lists it, padding the device assembly between the
    compile and the assembler step (mfmahaz.fix_asm). Returns what was padded.

    clang 24's new offload driver builds a host+device module with no device
    assembly step: clang-linker-wrapper generates the device code at link time.
    Such a command is planned again under --no-offload-new-driver, which has
    the step. A plan with no device assembly at all is a CompileError: a
    rebuild that pads nothing would look like a kernel with nothing to pad."""
    from . import mfmahaz  # noqa: PLC0415

    steps = _plan(cmd, tmp)
    if not any(_device_asm_step(a) for a in steps) and \
            any(os.path.basename(a[0]) == "clang-linker-wrapper" for a in steps):
        steps = _plan([*cmd, "--no-offload-new-driver"], tmp)
    if not any(_device_asm_step(a) for a in steps):
        raise CompileError("MFMA padding rebuild: hipcc's pipeline for this build has no device "
                           "assembly step to pad:\n" + "\n".join(" ".join(a) for a in steps))
    padded: List[str] = []
    for argv in steps:
        r = subprocess.run(argv, cwd=tmp, capture_output=True, text=True)
        if r.returncode != 0:
            raise CompileError(f"MFMA padding rebuild: {argv[0]} failed:\n{r.stderr}")
        if _device_asm_step(argv):
            asm = tmp / argv[argv.index("-o") + 1]
            text, said = mfmahaz.fix_asm(asm.read_text(), arch)
            asm.write_text(text)
            padded += said
    return padded


def mfma_gate_status() -> List[str]:
    """The MFMA gate for `python3 -m hk.integration --status`: its mode, whether this
    toolchain's compiler needs it, and the cached builds it padded (each kernel, how many
    places; the places themselves are in that build's MFMAHAZ and build.log)."""
    from . import cocheck, mfmahaz  # noqa: PLC0415

    fixed = mfmahaz.compiler_fixed(cocheck._tool("llc"))
    lines = [f"MFMA gate: HK_MFMA_GATE={_mfma_mode()}; this compiler "
             + {True: "pads every path (llvm/llvm-project#225989): not checked",
                False: "lacks llvm/llvm-project#225989: CDNA builds are checked",
                None: "could not be probed (no llc): CDNA builds are checked"}[fixed]]
    padded = []
    for mark in sorted(cache_dir().glob(f"*/{_MFMA_MARK}")):
        text = mark.read_text().splitlines()
        if text and text[0].startswith("padded "):
            kernels = sorted({t.split(":", 1)[0] for t in text[1:] if ":" in t})
            padded.append(f"  {mark.parent.name[:12]} {', '.join(kernels)}: "
                          f"{text[0].split()[1]} place(s)")
    lines.append(f"  {len(padded)} cached build(s) padded" + (":" if padded else ""))
    return lines + padded


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
    instantiation spills, if its code reads registers an LDS load is still
    filling (the wait gate, `_wait_gate`), or, on CDNA, if it reads or
    overwrites an MFMA result too soon on some path (the MFMA gate,
    `_mfma_found`) unless HK_MFMA_GATE=fix -- the default -- can pad it. A cache entry
    from before a gate is checked on its first hit.
    """
    if kind not in ("so", "co"):
        raise ValueError(f"build kind {kind!r}; 'so' or 'co'")
    target = get_target(arch)
    base = cache_key(source, target, extra_flags, kind)
    key_fl = (flags(target, _KEY_MODULE, extra_flags) if kind == "so"
              else codeobj_flags(target, extra_flags))
    _BUILT.add(hashlib.sha256("\0".join(
        [source, *(f for f in key_fl if not f.startswith("-I"))]).encode()).hexdigest())
    gates = dict(allow_scratch=allow_scratch, max_vgprs=max_vgprs, min_occupancy=min_occupancy)

    deps = _read_manifest(base)
    if deps is not None:
        hit = _hit(cache_dir() / _full_key(base, headers_digest(deps)), target, kind, gates)
        if hit is not None:
            return hit

    _jit_notice(name)
    for _ in range(3):
        got = _build_once(source, target, name, extra_flags, kind, gates, verbose, base, key_fl)
        if got is not None:
            return got
    raise CompileError(f"the headers kernel {name!r} includes kept changing while it compiled "
                       f"(three times); nothing was cached. Build again once they are written.")


def _hit(out: Path, target: Target, kind: str, gates: dict) -> Optional[Build]:
    """The cache entry at `out`, re-gated, or None (none there, or one to rebuild)."""
    if (hit := _load_cached(out, target)) is None:
        return None
    if not (out / _MFMA_MARK).exists() and _mfma_mode() == "fix" and \
            _mfma_found(hit.so_path, kind, target.arch):
        # built unpadded (the gate off, or before it): rebuilt here, padded, in its place
        shutil.rmtree(out, ignore_errors=True)
        return None
    # Re-gate on a hit. The key covers the source and the flags but not the
    # thresholds, so the same bytes can be asked for by one kernel with no
    # occupancy requirement and by another with a strict one; whoever
    # compiled first must not decide for the second. Cheap -- the numbers
    # come from the cached log, not from hipcc. Entries whose log no longer
    # parses gate on nothing, which is what the OK marker already attests.
    if hit.kernels:
        _note_unused_frames(hit.kernels, hit.so_path, kind)
        resources.gate(hit.kernels, **gates)
    if not (out / _WAIT_MARK).exists() and _wait_gate(hit.so_path, kind):
        # built before the wait gate, or with it off: checked once, then marked
        try:
            (out / _WAIT_MARK).write_text("lds copies checked\n")
        except OSError:
            pass
    if not (out / _MFMA_MARK).exists():
        # built before the MFMA gate, or with it off: a refusal names the fix (a rebuild)
        found = _mfma_found(hit.so_path, kind, target.arch)
        if found and _mfma_mode() == "report":
            print(f"hk: {hit.module_name}: {_mfma_refusal(found)}", file=sys.stderr)
        elif found:
            raise ResourceError(f"{_mfma_refusal(found)}\nCached build: {out}")
        elif found is not None:
            try:
                (out / _MFMA_MARK).write_text("mfma hazards checked\n")
            except OSError:
                pass
    return hit


def _build_once(source: str, target: Target, name: str, extra_flags: Sequence[str], kind: str,
                gates: dict, verbose: bool, base: str, key_fl: Sequence[str]) -> Optional[Build]:
    """List the headers, key, compile, gate, publish. None if a header changed
    while it compiled: the binary then matches neither version's key."""
    # Build in a scratch directory and move into place, so a killed or failing
    # compile never leaves a half-written .so that the next run would import.
    tmp = Path(tempfile.mkdtemp(prefix="hk-build-", dir=cache_dir()))
    moved = False
    try:
        src = tmp / "kernel.cpp"
        src.write_text(source)
        try:
            deps = _scan_deps(key_fl, src, kind, tmp)
        except CompileError as e:
            raise CompileError(f"kernel {name!r}: {e}\nSource kept at {_keep(tmp, base)}") from None
        digest = headers_digest(deps)
        key = _full_key(base, digest)
        out = cache_dir() / key
        if (hit := _hit(out, target, kind, gates)) is not None:
            _write_manifest(base, deps)
            return hit

        module_name = f"hk_{name}_{key[:12]}"
        so_name = module_name + ext_suffix() if kind == "so" else CODEOBJ
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

        kernels = resources.parse(log, target, compiler=compiler_version())
        _note_unused_frames(kernels, tmp / so_name, kind, tmp / "build.log")
        try:
            resources.gate(kernels, **gates)
            checked = _wait_gate(tmp / so_name, kind)
            mfma = _mfma_found(tmp / so_name, kind, target.arch)
            padded: List[str] = []
            if mfma and _mfma_mode() == "fix":
                padded = _pad_mfma_hazards(cmd, tmp, target.arch)
                line = (f"hk: {name}: HK_MFMA_GATE=fix padded {len(padded)} MFMA hazard(s):\n"
                        + "".join(f"  {p}\n" for p in padded))
                print(line, file=sys.stderr, end="")
                with open(tmp / "build.log", "a") as f:
                    f.write(line)
                mfma = _mfma_found(tmp / so_name, kind, target.arch)
            if mfma and _mfma_mode() == "report":
                line = f"hk: {name}: {_mfma_refusal(mfma)}\n"
                print(line, file=sys.stderr, end="")
                with open(tmp / "build.log", "a") as f:
                    f.write(line)
                mfma = None
            elif mfma:
                raise _mfma_refusal(mfma)
        except ResourceError as e:
            raise ResourceError(
                f"{e}\n\nGenerated source kept at {_keep(tmp, key)}"
            ) from None

        if headers_digest(deps) != digest:
            return None
        if checked:
            (tmp / _WAIT_MARK).write_text("lds copies checked\n")
        if mfma is not None:
            (tmp / _MFMA_MARK).write_text(
                f"padded {len(padded)}\n" + "".join(f"{p}\n" for p in padded) if padded
                else "mfma hazards checked\n")
        (tmp / "deps").write_text("\n".join(deps) + "\n")
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
            _write_manifest(base, deps)
            return _load_cached(out, target)
        _write_manifest(base, deps)
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
