from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("SESSION_STORE_BACKEND", "memory")
os.environ.setdefault("ACTION_STORE_BACKEND", "memory")
os.environ.setdefault("GRAPH_CHECKPOINTER_BACKEND", "memory")
