"""Train and evaluate the standalone masked-card predictor from normalized decks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from ygoai.deck_prediction.data import Deck, load_decks_jsonl, make_samples, split_decks
from ygoai.deck_prediction.model import DeckMaskPredictor


def evaluate(
    model: DeckMaskPredictor,
    samples: Sequence[dict[str, Any]],
    vocabulary_size: int,
) -> dict[str, float | int]:
    if not samples:
        return {"samples": 0, "main_total_mae": 0.0, "extra_total_mae": 0.0, "per_card_count_mae": 0.0}
    absolute_main = 0.0
    absolute_extra = 0.0
    absolute_card = 0.0
    for sample in samples:
        prediction = model.predict(sample["observed_main"], sample["observed_extra"])
        target_main = Counter({int(code): int(count) for code, count in sample["target_main"].items()})
        target_extra = Counter({int(code): int(count) for code, count in sample["target_extra"].items()})
        absolute_main += abs(sum(prediction["main"].values()) - sum(target_main.values()))
        absolute_extra += abs(sum(prediction["extra"].values()) - sum(target_extra.values()))
        all_main = set(prediction["main"]) | set(target_main)
        all_extra = set(prediction["extra"]) | set(target_extra)
        absolute_card += sum(abs(prediction["main"].get(code, 0.0) - target_main.get(code, 0)) for code in all_main)
        absolute_card += sum(abs(prediction["extra"].get(code, 0.0) - target_extra.get(code, 0)) for code in all_extra)
    count = len(samples)
    return {
        "samples": count,
        "main_total_mae": absolute_main / count,
        "extra_total_mae": absolute_extra / count,
        "per_card_count_mae": absolute_card / (count * max(1, 2 * vocabulary_size)),
    }


def train(
    *,
    dataset: str,
    output: str,
    seed: int = 0,
    validation_fraction: float = 0.2,
    samples_per_deck: int = 64,
    neighbors: int = 8,
    minimum_visible: int = 1,
    maximum_visible: int = 10,
) -> dict[str, Any]:
    decks = load_decks_jsonl(dataset)
    if not decks:
        raise ValueError(f"no decks found in {dataset}")
    train_decks, validation_decks = split_decks(decks, validation_fraction, seed)
    train_samples = make_samples(
        train_decks, samples_per_deck=samples_per_deck, seed=seed,
        minimum=minimum_visible, maximum=maximum_visible,
    )
    validation_samples = make_samples(
        validation_decks, samples_per_deck=samples_per_deck, seed=seed + 1,
        minimum=minimum_visible, maximum=maximum_visible,
    )
    model = DeckMaskPredictor.fit(train_samples, neighbors=neighbors)
    vocabulary = {code for deck in decks for code in (*deck.main, *deck.extra)}
    metrics = evaluate(model, validation_samples, len(vocabulary))
    manifest = {
        "schema_version": 1,
        "dataset": str(dataset),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "samples_per_deck": samples_per_deck,
        "neighbors": neighbors,
        "minimum_visible": minimum_visible,
        "maximum_visible": maximum_visible,
        "train_deck_ids": [deck.deck_id for deck in train_decks],
        "validation_deck_ids": [deck.deck_id for deck in validation_decks],
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "validation_metrics": metrics,
    }
    model.save(output)
    metrics_path = Path(output).with_suffix(Path(output).suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/deck_prediction/decks.jsonl")
    parser.add_argument("--output", default="data/deck_prediction/model.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--samples-per-deck", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--minimum-visible", type=int, default=1)
    parser.add_argument("--maximum-visible", type=int, default=10)
    args = parser.parse_args(argv)
    manifest = train(
        dataset=args.dataset,
        output=args.output,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        samples_per_deck=args.samples_per_deck,
        neighbors=args.neighbors,
        minimum_visible=args.minimum_visible,
        maximum_visible=args.maximum_visible,
    )
    print(json.dumps({
        "train_decks": len(manifest["train_deck_ids"]),
        "validation_decks": len(manifest["validation_deck_ids"]),
        "train_samples": manifest["train_sample_count"],
        "validation_samples": manifest["validation_sample_count"],
        "validation_metrics": manifest["validation_metrics"],
        "model": args.output,
        "manifest": str(Path(args.output).with_suffix(Path(args.output).suffix + ".metrics.json")),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
