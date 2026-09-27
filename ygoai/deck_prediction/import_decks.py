"""Import local YDKs and a local Yu-Gi-Oh! Meta filtered-results JSON snapshot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable

from ygoai.deck_prediction.data import (
    META_SOURCE_URL,
    META_TOS_URL,
    OCG_JULY_2026_BANLIST_URL,
    CardCatalog,
    Deck,
    read_meta_snapshot,
    read_repository_decks,
    validate_and_deduplicate,
)


def _snapshot_paths(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Meta JSON snapshot not found: {path}")
    return paths


def import_decks(
    *,
    meta_json: Iterable[str],
    deck_dir: str,
    cards_db: str,
    output_dir: str,
    include_git_files: bool = True,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    catalog = CardCatalog(cards_db)
    candidates: list[Deck] = []
    rejected: list[dict] = []
    source_files = []

    for path in _snapshot_paths(meta_json):
        source_files.append(str(path))
        meta_decks, meta_rejected = read_meta_snapshot(path, catalog)
        candidates.extend(meta_decks)
        rejected.extend(meta_rejected)

    repository_decks, repository_rejected = read_repository_decks(
        deck_dir, include_git_files=include_git_files
    )
    candidates.extend(repository_decks)
    rejected.extend(repository_rejected)
    accepted, validation_rejected, duplicates = validate_and_deduplicate(candidates, catalog)
    rejected.extend(validation_rejected)

    deck_path = output / "decks.jsonl"
    with deck_path.open("w", encoding="utf-8") as stream:
        for deck in accepted:
            stream.write(json.dumps(deck.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    reason_counts = Counter(item["reason"] for item in rejected)
    report = {
        "schema_version": 1,
        "sources": {
            "meta": {
                "name": "Yu-Gi-Oh! Meta",
                "url": META_SOURCE_URL,
                "tos_url": META_TOS_URL,
                "filter": {"format": "OCG", "banlist": "effective 2026-07-01"},
                "banlist_source": OCG_JULY_2026_BANLIST_URL,
                "snapshot_files": source_files,
            },
            "repository_deck_dir": str(deck_dir),
            "cards_db": str(Path(cards_db)),
        },
        "counts": {
            "candidate_records": len(candidates),
            "accepted_unique_decks": len(accepted),
            "meta_records": sum(deck.source == "yugiohmeta" for deck in candidates),
            "repository_records": sum(deck.source == "repository" for deck in candidates),
            "duplicates_removed": len(duplicates),
            "rejected_records": len(rejected),
        },
        "rejected_by_reason": dict(sorted(reason_counts.items())),
        "duplicates": duplicates,
        "rejected": rejected,
        "accepted_decks": [
            {
                "deck_id": deck.deck_id,
                "name": deck.name,
                "source": deck.source,
                "main_size": len(deck.main),
                "extra_size": len(deck.extra),
            }
            for deck in accepted
        ],
        "data_file": str(deck_path),
    }
    (output / "import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "source-metadata.json").write_text(
        json.dumps(report["sources"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-json",
        action="append",
        default=[],
        help="local JSON export of the OCG / July 2026 banlist filtered Meta results; repeatable",
    )
    parser.add_argument("--deck-dir", default="assets/deck", help="local repository YDK directory")
    parser.add_argument("--cards-db", default="assets/locale/en/cards.cdb", help="local YGOPro cards.cdb")
    parser.add_argument("--output-dir", default="data/deck_prediction", help="ignored local output directory")
    parser.add_argument(
        "--no-git-deck-fallback",
        action="store_true",
        help="do not read tracked top-level YDK files missing from the working tree",
    )
    args = parser.parse_args(argv)
    report = import_decks(
        meta_json=args.meta_json,
        deck_dir=args.deck_dir,
        cards_db=args.cards_db,
        output_dir=args.output_dir,
        include_git_files=not args.no_git_deck_fallback,
    )
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    print(f"Wrote {report['data_file']}")
    print(f"Import report: {Path(args.output_dir) / 'import-report.json'}")
    if not report["counts"]["accepted_unique_decks"]:
        print("No usable decks were imported; see the rejection report.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
