"""Trivial Phase 0 smoke test: the package imports and exposes a version.

Real env/model/parity tests arrive in later phases (see PROJECTPLAN.md).
"""

from __future__ import annotations

import latentlab


def test_package_has_version() -> None:
    assert isinstance(latentlab.__version__, str)
    assert latentlab.__version__.count(".") == 2
