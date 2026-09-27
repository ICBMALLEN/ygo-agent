"""Deck parsing, card-resource validation, and reproducible mask sampling."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import subprocess
from typing import Any, Iterable, Iterator, Mapping, Sequence


META_SOURCE_URL = "https://www.yugiohmeta.com/top-decks"
META_TOS_URL = "https://www.yugiohmeta.com/terms-of-service"
OCG_JULY_2026_BANLIST_URL = "https://www.yugiohmeta.com/articles/news/july-1-2026/ocg-forbidden-list"
TOKEN_TYPE = 0x4000


@dataclass(frozen=True)
class Deck:
    """A normalized deck. Main and Extra contain one entry per card instance."""

    deck_id: str
    name: str
    source: str
    source_url: str | None
    main: tuple[int, ...]
    extra: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["main"] = list(self.main)
        value["extra"] = list(self.extra)
        return value


class CardCatalog:
    """Card IDs, names, and token flags read directly from a YGOPro cards.cdb."""

    def __init__(self, cards_db: str | Path):
        self.path = Path(cards_db)
        if not self.path.is_file():
            raise FileNotFoundError(f"Card database not found: {self.path}")
        try:
            conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
            rows = conn.execute(
                "SELECT d.id, d.type, COALESCE(t.name, '') "
                "FROM datas AS d LEFT JOIN texts AS t ON t.id = d.id"
            ).fetchall()
            conn.close()
        except sqlite3.Error as exc:
            raise ValueError(f"Could not read YGOPro card database {self.path}: {exc}") from exc

        self.types: dict[int, int] = {}
        self.names: dict[int, str] = {}
        self.codes_by_name: dict[str, set[int]] = {}
        for code, card_type, name in rows:
            code = int(code)
            self.types[code] = int(card_type)
            name = str(name).strip()
            self.names[code] = name
            if name:
                self.codes_by_name.setdefault(name.casefold(), set()).add(code)

    def resolve(self, value: Any) -> int:
        """Resolve a card passcode or an unambiguous localized card name."""
        if isinstance(value, bool):
            raise ValueError("boolean is not a card ID")
        if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
            return int(value)
        name = str(value).strip().casefold()
        matches = self.codes_by_name.get(name, set())
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            raise ValueError(f"card name is ambiguous in this database: {value!r}")
        raise ValueError(f"unknown card ID/name: {value!r}")

    def is_token(self, code: int) -> bool:
        return bool(self.types[code] & TOKEN_TYPE)


def parse_ydk(text: str) -> tuple[list[int], list[int], list[int]]:
    """Parse YDK sections and return (main, extra, side), preserving instances."""
    sections: dict[str, list[int]] = {"main": [], "extra": [], "side": []}
    current = "main"
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        marker = line.casefold()
        if marker == "#main":
            current = "main"
            continue
        if marker == "#extra":
            current = "extra"
            continue
        if marker in ("!side", "#side"):
            current = "side"
            continue
        if line.startswith(("#", "!")):
            continue
        if line.isdigit():
            sections[current].append(int(line))
        elif re.fullmatch(r"[+-]?\d+", line):
            raise ValueError(f"invalid card ID at line {line_no}: {line!r}")
    return sections["main"], sections["extra"], sections["side"]


def _first(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _quantity(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("card quantity must be an integer")
    try:
        quantity = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid card quantity: {value!r}") from exc
    if quantity < 0:
        raise ValueError("card quantity cannot be negative")
    return quantity


def parse_card_list(value: Any, catalog: CardCatalog) -> list[int]:
    """Parse repeated IDs, ID-to-count objects, or common deck-export card rows."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        # Compact maps such as {"89631139": 3} are common in JSON exports.
        if value and all(str(key).strip().isdigit() for key in value):
            result: list[int] = []
            for code, count in value.items():
                result.extend([catalog.resolve(code)] * _quantity(count))
            return result
        value = [value]
    if not isinstance(value, list):
        raise ValueError("card zone must be a list or a card-ID-to-count object")

    result = []
    for entry in value:
        card_value = entry
        count = 1
        if isinstance(entry, Mapping):
            count = _quantity(_first(entry, ("quantity", "count", "copies", "qty"), 1))
            card_value = _first(entry, ("code", "id", "card_id", "cardId", "passcode", "name"))
            nested = entry.get("card")
            if card_value is None and isinstance(nested, Mapping):
                card_value = _first(nested, ("code", "id", "card_id", "cardId", "passcode", "name"))
            elif card_value is None:
                card_value = nested
        if card_value is None:
            raise ValueError(f"could not find a card ID/name in row: {entry!r}")
        code = catalog.resolve(card_value)
        result.extend([code] * count)
    return result


def _banlist_matches(value: Any) -> bool:
    normalized = re.sub(r"[\s,]", "", str(value).casefold())
    return bool(
        re.search(r"2026[-/.]?0?7(?:[-/.]?0?1)?", normalized)
        or "july2026" in normalized
        or re.search(r"july1(?:st)?2026", normalized)
        or "2026年7月" in normalized
    )


def _deck_records(payload: Any) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, Mapping):
        raise ValueError("Meta snapshot must contain a JSON object or array")
    records = _first(payload, ("decks", "results", "items"))
    if records is None:
        records = [payload]
    if not isinstance(records, list) or any(not isinstance(record, Mapping) for record in records):
        raise ValueError("snapshot 'decks'/'results' field must be an array of objects")
    return records, payload


def read_meta_snapshot(path: str | Path, catalog: CardCatalog) -> tuple[list[Deck], list[dict[str, Any]]]:
    """Read a JSON export of Yu-Gi-Oh! Meta's OCG / July 2026 filtered results.

    Rows may omit format/banlist fields when the file itself is an export of the
    already-filtered results. If those fields are present, mismatches are rejected.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records, metadata = _deck_records(payload)
    top_format = _first(metadata, ("format", "game_format"))
    top_banlist = _first(metadata, ("banlist", "ban_list", "banlist_id"))
    top_url = _first(metadata, ("source_url", "url"), META_SOURCE_URL)
    result: list[Deck] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        name = str(_first(record, ("name", "deck_name", "archetype"), f"Meta deck {index + 1}"))
        deck_label = str(_first(record, ("id", "deck_id", "url"), f"row-{index + 1}"))
        row_format = _first(record, ("format", "game_format"), top_format)
        row_banlist = _first(record, ("banlist", "ban_list", "banlist_id"), top_banlist)
        if row_format is not None and "ocg" not in str(row_format).casefold():
            rejected.append({"deck": deck_label, "name": name, "reason": "format_not_ocg"})
            continue
        if row_banlist is not None and not _banlist_matches(row_banlist):
            rejected.append({"deck": deck_label, "name": name, "reason": "banlist_not_2026_07", "banlist": str(row_banlist)})
            continue
        try:
            main = parse_card_list(_first(record, ("main", "main_deck", "mainDeck")), catalog)
            extra = parse_card_list(_first(record, ("extra", "extra_deck", "extraDeck")), catalog)
        except (TypeError, ValueError) as exc:
            rejected.append({"deck": deck_label, "name": name, "reason": "unparseable_card_list", "detail": str(exc)})
            continue
        source_url = _first(record, ("source_url", "url", "deck_url"), top_url)
        result.append(
            Deck(
                deck_id=f"meta:{deck_label}",
                name=name,
                source="yugiohmeta",
                source_url=str(source_url) if source_url else None,
                main=tuple(main),
                extra=tuple(extra),
            )
        )
    return result, rejected


def _repository_deck_paths(deck_dir: str | Path, include_git_files: bool) -> Iterator[tuple[str, str]]:
    """Yield (display path, text) for top-level local YDKs and removed tracked YDKs."""
    deck_dir = Path(deck_dir)
    seen: set[str] = set()
    if deck_dir.is_dir():
        for path in sorted(deck_dir.glob("*.ydk")):
            seen.add(path.name)
            yield str(path), path.read_text(encoding="utf-8", errors="replace")
    if not include_git_files:
        return
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=deck_dir if deck_dir.is_dir() else deck_dir.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        root = Path(top)
        relative_dir = deck_dir.resolve().relative_to(root).as_posix()
        tracked = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", relative_dir],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return
    prefix = relative_dir.rstrip("/") + "/"
    for item in sorted(tracked):
        relative_item = item[len(prefix):] if item.startswith(prefix) else item
        if "/" in relative_item or not relative_item.casefold().endswith(".ydk"):
            continue
        name = Path(relative_item).name
        if name in seen:
            continue
        raw = subprocess.run(
            ["git", "show", f"HEAD:{item}"], cwd=root, capture_output=True, text=True
        )
        if raw.returncode == 0:
            yield item, raw.stdout


def read_repository_decks(
    deck_dir: str | Path, *, include_git_files: bool = True
) -> tuple[list[Deck], list[dict[str, Any]]]:
    decks: list[Deck] = []
    rejected: list[dict[str, Any]] = []
    for display_path, content in _repository_deck_paths(deck_dir, include_git_files):
        name = Path(display_path).stem
        if name.startswith("_"):
            rejected.append({"deck": display_path, "name": name, "reason": "token_deck_file"})
            continue
        try:
            main, extra, _side = parse_ydk(content)
        except ValueError as exc:
            rejected.append({"deck": display_path, "name": name, "reason": "invalid_ydk", "detail": str(exc)})
            continue
        decks.append(
            Deck(
                deck_id=f"repo:{name}",
                name=name,
                source="repository",
                source_url=None,
                main=tuple(main),
                extra=tuple(extra),
            )
        )
    return decks, rejected


def deck_fingerprint(deck: Deck) -> str:
    """Fingerprint zone-specific card counts, ignoring ordering and Side Deck."""
    main = sorted(Counter(deck.main).items())
    extra = sorted(Counter(deck.extra).items())
    payload = json.dumps({"main": main, "extra": extra}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_and_deduplicate(
    decks: Iterable[Deck], catalog: CardCatalog
) -> tuple[list[Deck], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject missing/token/empty decks and deduplicate zone composition."""
    accepted: list[Deck] = []
    rejected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    fingerprints: dict[str, Deck] = {}
    for deck in decks:
        missing = sorted({code for code in (*deck.main, *deck.extra) if code not in catalog.types})
        if missing:
            rejected.append({"deck": deck.deck_id, "name": deck.name, "reason": "unsupported_card_ids", "codes": missing})
            continue
        token_codes = sorted({code for code in (*deck.main, *deck.extra) if catalog.is_token(code)})
        if token_codes:
            rejected.append({"deck": deck.deck_id, "name": deck.name, "reason": "token_cards_in_deck", "codes": token_codes})
            continue
        if not deck.main:
            rejected.append({"deck": deck.deck_id, "name": deck.name, "reason": "empty_main_deck"})
            continue
        fingerprint = deck_fingerprint(deck)
        previous = fingerprints.get(fingerprint)
        if previous is not None:
            duplicates.append({"kept": previous.deck_id, "duplicate": deck.deck_id, "name": deck.name, "fingerprint": fingerprint})
            continue
        fingerprints[fingerprint] = deck
        accepted.append(deck)
    return accepted, rejected, duplicates


def load_decks_jsonl(path: str | Path) -> list[Deck]:
    decks = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            decks.append(
                Deck(
                    deck_id=str(value["deck_id"]),
                    name=str(value["name"]),
                    source=str(value["source"]),
                    source_url=value.get("source_url"),
                    main=tuple(int(code) for code in value["main"]),
                    extra=tuple(int(code) for code in value["extra"]),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid normalized deck on line {line_no}: {exc}") from exc
    return decks


def split_decks(
    decks: Sequence[Deck], validation_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Deck], list[Deck]]:
    """Split by whole deck with stable IDs and no train/validation overlap."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    ordered = sorted(decks, key=lambda deck: deck.deck_id)
    if len(ordered) < 2 or validation_fraction == 0:
        return ordered, []
    ranked = sorted(
        ordered,
        key=lambda deck: hashlib.sha256(f"{seed}:{deck.deck_id}".encode("utf-8")).digest(),
    )
    validation_count = min(len(ordered) - 1, max(1, int(round(len(ordered) * validation_fraction))))
    validation_ids = {deck.deck_id for deck in ranked[:validation_count]}
    train = [deck for deck in ordered if deck.deck_id not in validation_ids]
    validation = [deck for deck in ordered if deck.deck_id in validation_ids]
    return train, validation


def sample_mask(deck: Deck, rng: random.Random, minimum: int = 1, maximum: int = 10) -> dict[str, Any]:
    """Hide card instances without replacement; return zone-specific residual counts."""
    instances = [("main", code) for code in deck.main] + [("extra", code) for code in deck.extra]
    instances.sort()
    if not instances:
        raise ValueError(f"deck has no Main/Extra instances: {deck.deck_id}")
    if minimum < 1 or maximum < minimum:
        raise ValueError("mask size bounds must satisfy 1 <= minimum <= maximum")
    upper = min(maximum, len(instances))
    count = rng.randint(min(minimum, upper), upper)
    visible = rng.sample(instances, count)
    visible_main = Counter(code for zone, code in visible if zone == "main")
    visible_extra = Counter(code for zone, code in visible if zone == "extra")
    target_main = Counter(deck.main) - visible_main
    target_extra = Counter(deck.extra) - visible_extra
    return {
        "deck_id": deck.deck_id,
        "observed_main": sorted(code for zone, code in visible if zone == "main"),
        "observed_extra": sorted(code for zone, code in visible if zone == "extra"),
        "target_main": dict(sorted(target_main.items())),
        "target_extra": dict(sorted(target_extra.items())),
    }


def make_samples(
    decks: Sequence[Deck], samples_per_deck: int, seed: int, minimum: int = 1, maximum: int = 10
) -> list[dict[str, Any]]:
    if samples_per_deck < 1:
        raise ValueError("samples_per_deck must be positive")
    rng = random.Random(seed)
    return [
        sample_mask(deck, rng, minimum, maximum)
        for deck in sorted(decks, key=lambda value: value.deck_id)
        for _ in range(samples_per_deck)
    ]
