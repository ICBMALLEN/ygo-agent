"""Command-line inference entry point for the standalone deck-mask predictor."""

from __future__ import annotations

import argparse
import json
from typing import Any

from ygoai.deck_prediction.model import DeckMaskPredictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/deck_prediction/model.json")
    parser.add_argument("--main", type=int, nargs="*", default=[], help="visible Main Deck card IDs")
    parser.add_argument("--extra", type=int, nargs="*", default=[], help="visible Extra Deck card IDs")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    model = DeckMaskPredictor.load(args.model)
    print(json.dumps(model.predict_top(args.main, args.extra, args.limit), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
