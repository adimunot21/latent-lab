"""The Python env must reproduce shared/fixtures/ exactly (regression guard).

The same fixtures gate the TypeScript port in web CI. If this test fails, the
env's dynamics changed: regenerate fixtures ONLY if the change is intentional,
and expect the TS port to need the same change (see the portability contract
in envs/two_rooms.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latentlab.envs.two_rooms import TwoRoomsConfig, TwoRoomsEnv

FIXTURES = Path(__file__).resolve().parents[2] / "shared" / "fixtures" / "two_rooms_parity.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    assert FIXTURES.exists(), "run: uv run python -m latentlab.envs.make_fixtures"
    return json.loads(FIXTURES.read_text())


def test_fixture_config_matches_current_default(payload: dict) -> None:
    import dataclasses

    assert payload["env_config"] == dataclasses.asdict(TwoRoomsConfig())


def test_all_cases_replay_exactly(payload: dict) -> None:
    for case in payload["cases"]:
        env = TwoRoomsEnv(config=TwoRoomsConfig(), seed=0)
        env.set_state(*case["start"])
        for i, (action, expected) in enumerate(zip(case["actions"], case["states"], strict=True)):
            state = env.step((action[0], action[1]))
            assert state == tuple(expected), (
                f"case '{case['name']}' step {i}: {state} != {tuple(expected)} (exact)"
            )


def test_frames_replay_byte_exact(payload: dict) -> None:
    env = TwoRoomsEnv(config=TwoRoomsConfig(), seed=0)
    for fixture in payload["frames"]:
        env.set_state(*fixture["state"])
        frame = env.render().flatten()
        expected = np.asarray(fixture["frame"], dtype=np.uint8)
        assert np.array_equal(frame, expected), f"frame '{fixture['name']}' differs"
