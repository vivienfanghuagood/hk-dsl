"""Every route that is on by default holds its engine contract in every production image we record
and in every engine version hk claims, judged from the recorded facts (tests/hk/ir/engine_facts/,
written by `python3 -m hk.integration.contract --record` inside the image) -- so a stand-in or a
check that changes is held to the real engines here, without any image in CI. A route whose
contract an image breaks today is listed in that fixture's `expected_refusals`, with the reason.

A refusal is per route (contract.apply_routes): the refused route stays off with its own warning,
and the image's other default-on routes are applied. The last test holds that against each
fixture's own facts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk.integration import contract, switches

FIXTURES = Path(__file__).resolve().parent / "engine_facts"
#: The module a route's engine is, in a fixture's `engines` (an image without it skips the route).
ENGINE_MODULE = {"vllm": "vllm", "sglang": "sglang", "sglang_diffusion": "sglang"}


def _fixtures():
    return sorted(FIXTURES.glob("*.json"))


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _installed(fx, engine):
    """The image has `engine`, and the fixture is recorded for it: `serves`, when present, names the
    engines a fixture holds hk to (an SGLang venv that also has a vLLM hk does not claim)."""
    mod = ENGINE_MODULE[engine]
    return (mod in fx.get("serves", fx["engines"])
            and not fx["engines"][mod].endswith("(version unknown)"))


def _cases():
    out = []
    for path in _fixtures():
        fx = _load(path)
        for r in switches.ROUTES.values():
            if r.default and fx["deployed_on"] in r.arches and r.engine in ENGINE_MODULE \
                    and _installed(fx, r.engine) and r.name in contract.CHECKS:
                out.append((path.stem, r.name))
    return out


def test_every_fixture_is_a_recorded_production_image():
    assert _fixtures(), "no engine facts recorded"
    for path in _fixtures():
        fx = _load(path)
        assert fx["image"].startswith("sha256:") and fx["image_note"], path.name
        assert fx["deployed_on"] in ("gfx1100", "gfx1201", "gfx950"), path.name
        assert set(fx["facts"]) == set(contract.CHECKS), \
            f"{path.name}: re-record it, the routes changed ({sorted(set(contract.CHECKS) ^ set(fx['facts']))})"
        for r in fx["expected_refusals"]:
            assert r in contract.CHECKS, f"{path.name}: unknown route {r}"
        assert set(fx.get("serves", ())) <= set(fx["engines"]), path.name
        assert fx["role"] in contract.ROLES and fx["label"], path.name     # the support matrix's
        assert "python" in fx["stack"] and "torch" in fx["stack"], path.name


@pytest.mark.parametrize("image,route", _cases(), ids=[f"{i}-{r}" for i, r in _cases()])
def test_a_default_on_route_holds_its_contract_in_every_recorded_image(image, route):
    fx = _load(FIXTURES / f"{image}.json")
    problems = contract.judge(route, fx["facts"][route])
    why = fx["expected_refusals"].get(route)
    if why is None:
        assert problems == [], (
            f"{route}'s contract breaks in {image} ({fx['image_note']}): {problems}. The route is on "
            f"by default there, and would be off. Fix the stand-in or the check, or -- if the image "
            f"really changed -- re-record it and list the refusal.")
    else:
        assert problems, (
            f"{route} now holds in {image}: drop it from that fixture's expected_refusals "
            f"(was: {why})")


def test_a_known_refusal_turns_off_only_its_route(monkeypatch):
    """apply_routes over a fixture's facts: each expected refusal is off, every other default-on
    route of that engine is applied."""
    seen = 0
    for path in _fixtures():
        fx = _load(path)
        for engine in {switches.ROUTES[r].engine for r in fx["expected_refusals"]}:
            if not _installed(fx, engine):
                continue
            on = [o.name for o in switches.ROUTES.values() if o.engine == engine and o.default
                  and fx["deployed_on"] in o.arches and o.name in contract.CHECKS]
            refused = [r for r in fx["expected_refusals"] if r in on]
            if not refused:
                continue
            for r in on:
                monkeypatch.setitem(contract.CHECKS, r,
                                    lambda r=r: contract.judge(r, fx["facts"][r]))
            _, applied, off = contract.apply_routes(
                engine, on, [(r, lambda r=r: f"{r} applied") for r in on])
            assert sorted(off) == sorted(refused), (path.name, off)
            assert applied == [r for r in on if r not in refused], (path.name, applied)
            seen += 1
    assert seen, "no fixture has an expected refusal of a default-on route: drop or rewrite this"
