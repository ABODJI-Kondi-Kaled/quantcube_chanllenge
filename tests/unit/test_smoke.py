from __future__ import annotations

from quantcube_challenge import __version__


def test_package_importable() -> None:
    assert __version__ == "0.1.0"
