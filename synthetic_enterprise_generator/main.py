"""Package-local entrypoint matching the requested project structure."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_enterprise_generator.cli import main


if __name__ == "__main__":
    main()
