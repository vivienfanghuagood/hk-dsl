"""The compile cache's key covers the headers a kernel's compile reads, and only those.

No hipcc: the compiler's dependency list (`_scan_deps`) and the compile itself are
stand-ins, so this pins the mechanism -- manifest, header digest, key -- in the IR tier.
tests/hk/codegen/test_compile.py runs the same with hipcc's own dependency list.
"""

from __future__ import annotations

import subprocess

import pytest

from hk.runtime import compile as C


@pytest.fixture
def cc(tmp_path, monkeypatch):
    """compile.build with a stand-in hipcc whose "binary" is its headers' contents.

    `scans` counts the dependency scans; `deps` is what the scan reports."""
    inc = tmp_path / "include"
    (inc / "hk").mkdir(parents=True)
    for name, text in (("kittens.cuh", "k"), ("hk/shim.cuh", "s1"), ("hk/inner.cuh", "i1"),
                       ("hk/other.cuh", "o1")):
        (inc / name).write_text(text)
    state = dict(scans=0, compiles=0,
                 deps=[str(inc / n) for n in ("kittens.cuh", "hk/shim.cuh", "hk/inner.cuh")])
    monkeypatch.setenv("HK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HK_WAIT_GATE", "0")
    monkeypatch.setenv("HK_MFMA_GATE", "off")
    monkeypatch.setattr(C, "include_dirs", lambda: [inc])
    monkeypatch.setattr(C, "hipcc", lambda: "/fake/hipcc")
    monkeypatch.setattr(C, "_hipcc_version", lambda: "hipcc x")
    monkeypatch.setattr(C, "rocm_path", lambda: tmp_path / "rocm")
    monkeypatch.setattr(C.resources, "parse", lambda log, target=None: [])
    monkeypatch.setattr(C, "_FILE_HASH", {})

    def scan(fl, src, kind, tmp):
        state["scans"] += 1
        return sorted(state["deps"])

    def run(cmd, **kw):
        state["compiles"] += 1
        out = cmd[cmd.index("-o") + 1]
        with open(out, "w") as f:
            f.write("".join(open(p).read() for p in sorted(state["deps"])))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(C, "_scan_deps", scan)
    monkeypatch.setattr(C.subprocess, "run", run)

    def build(src="// kernel\n"):
        b = C.build(src, "gfx950", name="k", kind="co")
        return b, b.so_path.read_text()
    state["build"], state["inc"] = build, inc
    return state


def test_an_edit_to_a_header_two_includes_deep_rebuilds_the_kernel(cc):
    """include/hk/cdna_agpr.cuh, which cdna.cuh includes, was in no key: an edit to it
    served the binary built from the old one, from a different-looking cache entry of
    the same name."""
    b0, bin0 = cc["build"]()
    assert not b0.cached and bin0 == "i1s1k"
    b1, _ = cc["build"]()
    assert b1.cached and b1.key == b0.key
    (cc["inc"] / "hk" / "inner.cuh").write_text("i2")
    b2, bin2 = cc["build"]()
    assert not b2.cached and b2.key != b0.key and bin2 == "i2s1k"


def test_an_edit_to_a_header_the_kernel_does_not_include_keeps_it_cached(cc):
    """The reason the key is per kernel and not one hash of the tree: a branch touching
    the CDNA shim must not compile CI's RDNA kernels cold (40-56 min, PRs #28-#31)."""
    b0, _ = cc["build"]()
    (cc["inc"] / "hk" / "other.cuh").write_text("o2")
    b1, _ = cc["build"]()
    assert b1.cached and b1.key == b0.key
    assert cc["compiles"] == 1


def test_a_hit_takes_no_scan_and_a_new_include_is_found_on_the_rebuild(cc):
    cc["build"]()
    assert cc["scans"] == 1
    cc["build"]()
    assert cc["scans"] == 1, "a cache hit must not run the preprocessor"
    # shim.cuh now includes other.cuh: the edit to shim.cuh misses, the rescan sees other.cuh
    (cc["inc"] / "hk" / "shim.cuh").write_text("s2")
    cc["deps"].append(str(cc["inc"] / "hk" / "other.cuh"))
    b, bin_ = cc["build"]()
    assert not b.cached and bin_ == "i1o1s2k"
    (cc["inc"] / "hk" / "other.cuh").write_text("o2")
    b2, bin2 = cc["build"]()
    assert not b2.cached and bin2 == "i1o2s2k"


def test_reverting_an_edit_finds_the_first_build_again(cc):
    b0, _ = cc["build"]()
    (cc["inc"] / "hk" / "inner.cuh").write_text("i2")
    cc["build"]()
    (cc["inc"] / "hk" / "inner.cuh").write_text("i1")
    b2, bin2 = cc["build"]()
    assert b2.key == b0.key and bin2 == "i1s1k"


def test_a_header_edited_while_it_compiles_is_not_cached_under_either_version(cc, monkeypatch):
    real = C.subprocess.run
    edits = iter(["i2"])

    def run(cmd, **kw):
        p = real(cmd, **kw)
        for text in edits:      # once: the header changes after the scan, during the compile
            (cc["inc"] / "hk" / "inner.cuh").write_text(text)
        return p

    monkeypatch.setattr(C.subprocess, "run", run)
    b, bin_ = cc["build"]()
    assert cc["compiles"] == 2 and bin_ == "i2s1k"
    assert b.key == C._full_key(C.cache_key("// kernel\n", C.get_target("gfx950"), (), "co"),
                                C.headers_digest(sorted(cc["deps"])))


def test_a_manifest_from_another_key_scheme_is_ignored(cc):
    b0, _ = cc["build"]()
    base = C.cache_key("// kernel\n", C.get_target("gfx950"), (), "co")
    m = C._manifest(base)
    assert m.read_text().split("\n")[0] == C._KEY_SCHEME
    m.write_text("some other scheme\n/nonexistent.cuh\n")
    assert C._read_manifest(base) is None
    b1, _ = cc["build"]()
    assert b1.cached and b1.key == b0.key       # rescanned, found, and the manifest rewritten
    assert C._read_manifest(base) == sorted(cc["deps"])


def test_the_dependency_file_parser():
    text = ("hkdeps: /a/kittens.cuh /b/with\\ space.cuh \\\n"
            "  /c/dollar$$.h \\\n  rel/x.cuh\n")
    assert C._parse_depfile(text) == ["/a/kittens.cuh", "/b/with space.cuh", "/c/dollar$.h",
                                      "rel/x.cuh"]


def test_torch_headers_are_keyed_by_torchs_version_not_their_contents(tmp_path, monkeypatch):
    """A torch upgraded in place keeps its include path; its version is what moves."""
    import os
    import sys
    import types

    fake = types.ModuleType("torch")
    fake.__version__ = "9.9.9"
    fake.__file__ = "/venv/lib/python3.12/site-packages/torch/__init__.py"
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.delitem(sys.modules, "pybind11", raising=False)
    monkeypatch.setattr(C, "rocm_path", lambda: tmp_path / "rocm")
    inc = "/venv/lib/python3.12/site-packages/torch/include"
    fl = [f"-I{inc}", "-O3"]
    assert any("torch 9.9.9" in v for v in C._external_versions(fl))
    assert C._external_versions(["-O3"]) == []
    assert os.path.realpath(inc) in C._versioned_dirs(fl)
    # a checkout that happens to sit under a directory named torch is not torch's
    assert C._external_versions(["-I/src/torch/hk-dsl/include"]) == []


def test_flags_hipcc_appends_from_its_environment_are_keyed(monkeypatch):
    monkeypatch.setattr(C, "_hipcc_version", lambda: "hipcc x")
    monkeypatch.setattr(C, "rocm_path", lambda: C.Path("/opt/rocm"))
    monkeypatch.setattr(C, "include_dirs", lambda: [C.Path("/inc")])
    t = C.get_target("gfx950")
    monkeypatch.delenv("HIPCC_COMPILE_FLAGS_APPEND", raising=False)
    k0 = C.cache_key("// k\n", t, (), "co")
    monkeypatch.setenv("HIPCC_COMPILE_FLAGS_APPEND", "-O0")
    assert C.cache_key("// k\n", t, (), "co") != k0
