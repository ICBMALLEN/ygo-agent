"""Download Yu-Gi-Oh! Meta's public OCG post-banlist deck results to local JSON."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SITE_URL = "https://www.yugiohmeta.com"
SOURCE_URL = f"{SITE_URL}/top-decks"
TOS_URL = f"{SITE_URL}/terms-of-service"
DEFAULT_BANLIST_DATE = "2026-07-01"
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def build_api_url(base_url: str, collection: str, params: Sequence[tuple[str, str]]) -> str:
    """Construct a Meta v1 endpoint URL using its public qs-style query syntax."""
    query = urlencode(params, doseq=True, safe="[]$")
    return f"{base_url.rstrip('/')}/api/v1/{collection}" + (f"?{query}" if query else "")


def request_json(
    url: str,
    *,
    timeout: float = 60.0,
    retries: int = 3,
    user_agent: str = "YGOAgentDeckMask/0.1 (local research dataset)",
) -> Any:
    """Fetch JSON politely, retrying transient API/network errors with backoff."""
    request = Request(url, headers={"Accept": "application/json", "User-Agent": user_agent})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Meta API returned HTTP {response.status}: {url}")
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS or attempt >= retries:
                endpoint = url.split("?", 1)[0]
                query_length = len(url.partition("?")[2])
                raise RuntimeError(
                    f"Meta API returned HTTP {exc.code}: {endpoint} (query length {query_length})"
                ) from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = max(float(retry_after), 0.0) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
        except (URLError, TimeoutError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Could not fetch Meta API JSON from {url}: {exc}") from exc
            delay = 2**attempt
        time.sleep(min(delay, 30.0))
    raise RuntimeError(f"Meta API request failed: {url}")


def fetch_decks(
    base_url: str,
    banlist_date: str,
    *,
    request: Callable[[str], Any] = request_json,
) -> list[dict[str, Any]]:
    """Request OCG legal deck records dated on/after the selected list date."""
    params = [
        ("created[$gte]", banlist_date),
        ("ocg", "true"),
        ("genesys", "false"),
        ("illegal", "false"),
        # Meta's page uses limit=0 for the complete filtered results, not zero rows.
        ("limit", "0"),
    ]
    payload = request(build_api_url(base_url, "top-decks", params))
    if not isinstance(payload, list):
        raise ValueError("Meta top-decks API response was not a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def _zone_entries(deck: Mapping[str, Any], zone: str) -> list[Mapping[str, Any]]:
    value = deck.get(zone)
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, Mapping)]


def card_database_ids(decks: Iterable[Mapping[str, Any]]) -> list[str]:
    ids = set()
    for deck in decks:
        for zone in ("main", "extra"):
            for entry in _zone_entries(deck, zone):
                card = entry.get("card")
                if isinstance(card, Mapping) and card.get("_id"):
                    ids.add(str(card["_id"]))
    return sorted(ids)


def fetch_card_mappings(
    base_url: str,
    meta_card_ids: Sequence[str],
    *,
    batch_size: int = 200,
    delay_seconds: float = 0.25,
    request: Callable[[str], Any] = request_json,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve Meta's card document IDs to Konami passcodes in bounded batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mapped: dict[str, dict[str, Any]] = {}
    batches = [meta_card_ids[i:i + batch_size] for i in range(0, len(meta_card_ids), batch_size)]
    for batch_number, batch in enumerate(batches, 1):
        if batch_number > 1 and delay_seconds > 0:
            time.sleep(delay_seconds)
        params = [(f"_id[$in][{index}]", card_id) for index, card_id in enumerate(batch)]
        params.append(("limit", "0"))
        payload = request(build_api_url(base_url, "cards", params))
        if not isinstance(payload, list):
            raise ValueError("Meta cards API response was not a JSON array")
        for card in payload:
            if isinstance(card, dict) and card.get("_id"):
                mapped[str(card["_id"])] = card
        if progress is not None:
            progress(batch_number, len(batches))
    return mapped


def _normalized_card(entry: Mapping[str, Any], mappings: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    card = entry.get("card")
    card = card if isinstance(card, Mapping) else {}
    meta_id = str(card.get("_id", ""))
    resolved = mappings.get(meta_id, {})
    name = str(resolved.get("name") or card.get("name") or "Unknown Meta card")
    quantity = entry.get("amount", entry.get("quantity", 1))
    result: dict[str, Any] = {
        "name": name,
        "quantity": int(quantity),
        "meta_card_id": meta_id or None,
    }
    passcode = resolved.get("konamiID")
    if passcode is not None and str(passcode).isdigit():
        result["code"] = int(passcode)
    return result


def normalize_decks(
    raw_decks: Sequence[Mapping[str, Any]],
    mappings: Mapping[str, Mapping[str, Any]],
    banlist_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Convert API card IDs/amounts to the importer's passcode/count schema."""
    result = []
    unmapped = []
    seen_unmapped: set[tuple[str, str]] = set()
    for deck in raw_decks:
        name_data = deck.get("deckType")
        name = name_data.get("name") if isinstance(name_data, Mapping) else None
        deck_id = str(deck.get("_id", ""))
        if not deck_id:
            continue
        normalized: dict[str, Any] = {
            "id": deck_id,
            "name": str(name or "Unnamed Meta deck"),
            "source_url": str(deck.get("url", "")),
            "format": "OCG",
            "banlist": banlist_date,
            "created": deck.get("created"),
            "main": [],
            "extra": [],
        }
        if normalized["source_url"] and normalized["source_url"].startswith("/"):
            normalized["source_url"] = SITE_URL + normalized["source_url"]
        for zone in ("main", "extra"):
            for entry in _zone_entries(deck, zone):
                card_record = _normalized_card(entry, mappings)
                normalized[zone].append(card_record)
                if "code" not in card_record:
                    key = (str(card_record.get("meta_card_id") or ""), card_record["name"])
                    if key not in seen_unmapped:
                        seen_unmapped.add(key)
                        unmapped.append({"meta_card_id": key[0], "name": key[1]})
        result.append(normalized)
    return result, unmapped


def _validate_api_filters(decks: Sequence[Mapping[str, Any]], banlist_date: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    included = []
    rejected = []
    for deck in decks:
        deck_id = str(deck.get("_id", "unknown"))
        created = str(deck.get("created", ""))[:10]
        event = deck.get("event") if isinstance(deck.get("event"), Mapping) else {}
        if deck.get("ocg") is not True or deck.get("genesys") is True:
            rejected.append({"deck_id": deck_id, "reason": "not_standard_ocg"})
        elif deck.get("illegal") is not False:
            rejected.append({"deck_id": deck_id, "reason": "not_banlist_legal"})
        elif event.get("format") not in (None, "OCG"):
            rejected.append({"deck_id": deck_id, "reason": "event_format_not_ocg"})
        elif not created or created < banlist_date:
            rejected.append({"deck_id": deck_id, "reason": "before_banlist_date_or_missing_date"})
        else:
            included.append(dict(deck))
    return included, rejected


def download(
    *,
    output_dir: str,
    banlist_date: str = DEFAULT_BANLIST_DATE,
    base_url: str = SITE_URL,
    batch_size: int = 50,
    delay_seconds: float = 0.25,
    request: Callable[[str], Any] = request_json,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    date.fromisoformat(banlist_date)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fetched = fetch_decks(base_url, banlist_date, request=request)
    raw_decks, rejected = _validate_api_filters(fetched, banlist_date)
    meta_ids = card_database_ids(raw_decks)
    mappings = fetch_card_mappings(
        base_url,
        meta_ids,
        batch_size=batch_size,
        delay_seconds=delay_seconds,
        request=request,
        progress=progress,
    )
    decks, unmapped = normalize_decks(raw_decks, mappings, banlist_date)
    raw_path = destination / "meta-api-raw.json"
    snapshot_path = destination / f"meta-ocg-post-{banlist_date}.json"
    report_path = destination / "meta-download-report.json"

    raw_path.write_text(json.dumps(fetched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    snapshot = {
        "source": "Yu-Gi-Oh! Meta",
        "source_url": SOURCE_URL,
        "tos_url": TOS_URL,
        "format": "OCG",
        "banlist": banlist_date,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "created_from_inclusive": banlist_date,
            "ocg": True,
            "genesys": False,
            "illegal": False,
            "api_endpoint": f"{base_url.rstrip('/')}/api/v1/top-decks",
        },
        "decks": decks,
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "source_url": SOURCE_URL,
        "banlist_date": banlist_date,
        "api_rows_received": len(fetched),
        "filtered_decks": len(decks),
        "card_documents_requested": len(meta_ids),
        "card_documents_resolved": len(mappings),
        "card_documents_missing": len(set(meta_ids) - set(mappings)),
        "card_passcodes_available": sum(
            card.get("konamiID") is not None and str(card.get("konamiID")).isdigit()
            for card in mappings.values()
        ),
        "unmapped_card_documents": len(unmapped),
        "unmapped_cards": unmapped,
        "rejected_api_rows": rejected,
        "raw_api_snapshot": str(raw_path),
        "normalized_snapshot": str(snapshot_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/deck_prediction/raw")
    parser.add_argument("--banlist-date", default=DEFAULT_BANLIST_DATE)
    parser.add_argument("--base-url", default=SITE_URL, help=argparse.SUPPRESS)
    parser.add_argument("--card-batch-size", type=int, default=50)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    args = parser.parse_args(argv)

    def show_progress(current: int, total: int) -> None:
        print(f"Resolving card IDs: batch {current}/{total}", flush=True)

    report = download(
        output_dir=args.output_dir,
        banlist_date=args.banlist_date,
        base_url=args.base_url,
        batch_size=args.card_batch_size,
        delay_seconds=args.delay_seconds,
        progress=show_progress,
    )
    print(json.dumps({key: value for key, value in report.items() if key not in ("unmapped_cards", "rejected_api_rows")}, ensure_ascii=False, sort_keys=True))
    if report["unmapped_cards"]:
        print(f"Unmapped Meta card documents: {len(report['unmapped_cards'])}; see {args.output_dir}/meta-download-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
