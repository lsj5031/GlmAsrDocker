"""Conftest: mock heavy ML imports so tests can import server.py without GPU/weights."""

import sys
from unittest.mock import MagicMock

# Create mock modules for all heavy dependencies that aren't installed in the test venv
_HEAVY_MODULES = [
    "torch",
    "torchaudio",
    "torchaudio.transforms",
    "transformers",
    "silero_vad",
    "ffmpeg",
    "soundfile",
    "pydub",
    "anyio",
]

for mod_name in _HEAVY_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Ensure nested attributes work for torchaudio.transforms
if "torchaudio.transforms" not in sys.modules:
    sys.modules["torchaudio.transforms"] = MagicMock()

# Mock numpy with a real-ish mock
import types

np_mock = MagicMock()
np_mock.ndarray = MagicMock
np_mock.zeros = lambda n, dtype=None: MagicMock(__len__=lambda s: n)
np_mock.array = lambda *a, **kw: MagicMock()
np_mock.float32 = "float32"
np_mock.concatenate = lambda *a, **kw: MagicMock()
np_mock.frombuffer = lambda *a, **kw: MagicMock()
sys.modules["numpy"] = np_mock
