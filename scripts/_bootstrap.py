"""Make ``python scripts/foo.py`` work from a clean checkout.

Importing this puts the repository root on ``sys.path`` so the scripts run
without ``pip install -e .`` first.  An installed package takes precedence, so
this is a convenience, not a hijack.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
