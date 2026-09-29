import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synth import make_beat_song  # noqa: E402


@pytest.fixture(scope="session")
def beat_song(tmp_path_factory):
    return make_beat_song(tmp_path_factory.mktemp("songs") / "click120.wav", bpm=120.0, seconds=20.0)
