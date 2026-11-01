"""Every route that is on by default holds its engine contract in every production image we record,
judged from the image's recorded facts (tests/hk/ir/engine_facts/, written by `python3 -m
hk.integration.contract --record` inside the image) -- so a stand-in or a check that changes is held
to the real engines here, without any image in CI. A route whose contract an image breaks today is
listed in that fixture's `expected_refusals`, with the reason.

Today a refusal is not per route: one broken contract makes the plugin leave the WHOLE engine stock
(contract.py, "all or nothing, per engine"), so an expected refusal means hk patches nothing in that
image while the route is on. The test says so in its message.
TODO(per-route contract policy, pending with the owner): if one route's refusal comes to turn only
that route off, assert here that the image's other default-on routes still apply.
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
    return not fx["engines"][ENGINE_MODULE[engine]].endswith("(version unknown)")


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


@pytest.mark.parametrize("image,route", _cases(), ids=[f"{i}-{r}" for i, r in _cases()])
def test_a_default_on_route_holds_its_contract_in_every_recorded_image(image, route):
    fx = _load(FIXTURES / f"{image}.json")
    problems = contract.judge(route, fx["facts"][route])
    why = fx["expected_refusals"].get(route)
    if why is None:
        assert problems == [], (
            f"{route}'s contract breaks in {image} ({fx['image_note']}): {problems}. With the route "
            f"on by default there, hk would patch NOTHING in that engine. Fix the stand-in or the "
            f"check, or -- if the image really changed -- re-record it.")
    else:
        assert problems, (
            f"{route} now holds in {image}: drop it from that fixture's expected_refusals "
            f"(was: {why})")
        # Today one refusal leaves the whole engine stock (all or nothing, per engine).


def test_a_known_refusal_is_said_as_a_whole_engine_refusal():
    for path in _fixtures():
        fx = _load(path)
        for r, why in fx["expected_refusals"].items():
            route = switches.ROUTES[r]
            if not (route.default and fx["deployed_on"] in route.arches):
                continue
            others = [o.name for o in switches.ROUTES.values()
                      if o.engine == route.engine and o.default and fx["deployed_on"] in o.arches
                      and o.name != r]
            line = contract.refuse(route.engine, [f"{r}: {p}" for p in
                                                  contract.judge(r, fx["facts"][r])])
            assert "hk patched NOTHING in this engine" in line, (
                f"{path.name}: {r} refuses, and today that takes {others} off with it")
