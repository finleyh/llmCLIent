"""Entry point: launch the REPL."""

from __future__ import annotations

from .config import Config
from .repl import Repl


def main() -> None:
    cfg = Config.load()
    Repl(cfg).run()


if __name__ == "__main__":
    main()
