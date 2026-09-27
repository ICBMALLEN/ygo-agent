import json
from collections import Counter
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest

from ygoai.deck_prediction.data import (
    CardCatalog,
    Deck,
    parse_ydk,
    read_meta_snapshot,
    sample_mask,
    split_decks,
    validate_and_deduplicate,
)
from ygoai.deck_prediction.download_meta import download
from ygoai.deck_prediction.import_decks import import_decks
from ygoai.deck_prediction.model import DeckMaskPredictor
from ygoai.deck_prediction.train import train


def make_catalog(path: Path) -> CardCatalog:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE datas(id INTEGER PRIMARY KEY, type INTEGER);"
        "CREATE TABLE texts(id INTEGER PRIMARY KEY, name TEXT);"
    )
    cards = [(101, 1, "Alpha"), (102, 1, "Beta"), (103, 1, "Gamma"), (900, 0x4000, "Token")]
    connection.executemany("INSERT INTO datas(id, type) VALUES (?, ?)", [(code, typ) for code, typ, _ in cards])
    connection.executemany("INSERT INTO texts(id, name) VALUES (?, ?)", [(code, name) for code, _, name in cards])
    connection.commit()
    connection.close()
    return CardCatalog(path)


class DeckPredictionTests(unittest.TestCase):
    def test_ydk_zones_ignore_side(self):
        main, extra, side = parse_ydk(
            "#created by test\n#main\n101\n101\n102\n#extra\n103\n!side\n900\n"
        )
        self.assertEqual(main, [101, 101, 102])
        self.assertEqual(extra, [103])
        self.assertEqual(side, [900])

    def test_instance_mask_is_reproducible_and_zone_safe(self):
        deck = Deck("deck-a", "Deck A", "test", None, (101, 101, 102), (103, 103))
        first = sample_mask(deck, random.Random(17))
        second = sample_mask(deck, random.Random(17))
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first["observed_main"]) + len(first["observed_extra"]), 1)
        self.assertLessEqual(len(first["observed_main"]) + len(first["observed_extra"]), 10)
        self.assertEqual(
            Counter(deck.main),
            Counter(first["observed_main"]) + Counter(first["target_main"]),
        )
        self.assertEqual(
            Counter(deck.extra),
            Counter(first["observed_extra"]) + Counter(first["target_extra"]),
        )
        self.assertFalse((Counter(first["observed_main"]) - Counter(deck.main)))
        self.assertFalse((Counter(first["observed_extra"]) - Counter(deck.extra)))

    def test_dedup_uses_zone_composition_and_rejects_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = make_catalog(Path(temporary) / "cards.cdb")
            first = Deck("meta:1", "One", "yugiohmeta", None, (101, 102, 101), (103,))
            same = Deck("repo:two", "Two", "repository", None, (102, 101, 101), (103,))
            token = Deck("repo:token", "Token", "repository", None, (101, 900), ())
            unsupported = Deck("meta:bad", "Bad", "yugiohmeta", None, (777, 101), ())
            accepted, rejected, duplicates = validate_and_deduplicate(
                [first, same, token, unsupported], catalog
            )
        self.assertEqual([deck.deck_id for deck in accepted], ["meta:1"])
        self.assertEqual(len(duplicates), 1)
        self.assertEqual({item["reason"] for item in rejected}, {"token_cards_in_deck", "unsupported_card_ids"})

    def test_meta_import_filters_and_reports_source_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = make_catalog(root / "cards.cdb")
            snapshot = root / "filtered.json"
            snapshot.write_text(json.dumps({
                "format": "OCG",
                "banlist": "2026-07-01",
                "source_url": "https://www.yugiohmeta.com/top-decks",
                "decks": [
                    {"id": "one", "name": "One", "banlist": "July 1st, 2026", "main": [101, 101, 102], "extra": [103], "side": [900]},
                    {"id": "same", "name": "Same", "main": [102, 101, 101], "extra": [103], "side": []},
                    {"id": "tcg", "format": "TCG", "name": "Wrong format", "main": [101], "extra": []},
                    {"id": "old", "banlist": "2026-01-01", "name": "Wrong list", "main": [101], "extra": []},
                    {"id": "unknown", "name": "Unknown", "main": [777], "extra": []},
                    {"id": "token", "name": "Token", "main": [101, 900], "extra": []},
                ],
            }), encoding="utf-8")
            output = root / "output"
            report = import_decks(
                meta_json=[str(snapshot)], deck_dir=str(root / "no-decks"),
                cards_db=str(catalog.path), output_dir=str(output), include_git_files=False,
            )
            imported = [json.loads(line) for line in (output / "decks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(report["counts"]["accepted_unique_decks"], 1)
        self.assertEqual(report["counts"]["duplicates_removed"], 1)
        self.assertEqual(len(imported[0]["main"]), 3)
        self.assertEqual(len(imported[0]["extra"]), 1)
        self.assertEqual(report["sources"]["meta"]["filter"]["banlist"], "effective 2026-07-01")
        self.assertEqual(
            {item["reason"] for item in report["rejected"]},
            {"format_not_ocg", "banlist_not_2026_07", "unsupported_card_ids", "token_cards_in_deck"},
        )

    def test_meta_downloader_fetches_filters_maps_and_keeps_raw_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_decks = [
                {
                    "_id": "meta-deck-1", "created": "2026-07-05T11:00:00.000Z",
                    "ocg": True, "genesys": False, "illegal": False,
                    "event": {"format": "OCG"}, "url": "/event/alpha/player/deck-id/",
                    "deckType": {"name": "Alpha"},
                    "main": [
                        {"card": {"_id": "meta-card-101", "name": "Alpha"}, "amount": 2},
                        {"card": {"_id": "meta-card-missing", "name": "Unmapped Card"}, "amount": 1},
                    ],
                    "extra": [{"card": {"_id": "meta-card-103", "name": "Gamma"}, "amount": 1}],
                    "side": [{"card": {"_id": "meta-card-side", "name": "Side"}, "amount": 1}],
                },
                {
                    "_id": "old-deck", "created": "2026-06-30T11:00:00.000Z",
                    "ocg": True, "genesys": False, "illegal": False,
                },
            ]
            card_docs = {
                "meta-card-101": {"_id": "meta-card-101", "name": "Alpha", "konamiID": "101"},
                "meta-card-103": {"_id": "meta-card-103", "name": "Gamma", "konamiID": "103"},
            }
            requested = []

            def fake_request(url):
                requested.append(url)
                if "/top-decks?" in url:
                    return raw_decks
                ids = [key for key in card_docs if key in url]
                return [card_docs[key] for key in ids]

            report = download(
                output_dir=str(root / "raw"), banlist_date="2026-07-01",
                batch_size=1, delay_seconds=0, request=fake_request,
            )
            snapshot = json.loads(Path(report["normalized_snapshot"]).read_text(encoding="utf-8"))
            raw = json.loads(Path(report["raw_api_snapshot"]).read_text(encoding="utf-8"))
        self.assertEqual(report["api_rows_received"], 2)
        self.assertEqual(report["filtered_decks"], 1)
        self.assertEqual(report["card_documents_requested"], 3)
        self.assertEqual(report["unmapped_card_documents"], 1)
        self.assertEqual(snapshot["decks"][0]["main"][0]["code"], 101)
        self.assertEqual(snapshot["decks"][0]["main"][1]["name"], "Unmapped Card")
        self.assertEqual(snapshot["decks"][0]["source_url"], "https://www.yugiohmeta.com/event/alpha/player/deck-id/")
        self.assertEqual(len(raw), 2)
        self.assertTrue(any("created[$gte]=2026-07-01" in url for url in requested))
        self.assertTrue(any("ocg=true" in url and "illegal=false" in url for url in requested))

    def test_deck_split_is_stable_and_disjoint(self):
        decks = [Deck(f"deck-{i}", str(i), "test", None, (101 + i,), ()) for i in range(8)]
        train_a, valid_a = split_decks(decks, 0.25, seed=23)
        train_b, valid_b = split_decks(decks, 0.25, seed=23)
        self.assertEqual([d.deck_id for d in train_a], [d.deck_id for d in train_b])
        self.assertEqual([d.deck_id for d in valid_a], [d.deck_id for d in valid_b])
        self.assertFalse({d.deck_id for d in train_a} & {d.deck_id for d in valid_a})

    def test_fit_save_load_and_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "decks.jsonl"
            decks = [
                Deck(f"deck-{i}", f"Deck {i}", "test", None, (101 + i, 101 + i, 102), (103,))
                for i in range(6)
            ]
            dataset.write_text("".join(json.dumps(deck.to_dict()) + "\n" for deck in decks), encoding="utf-8")
            model_path = root / "model.json"
            manifest = train(
                dataset=str(dataset), output=str(model_path), seed=4,
                validation_fraction=0.33, samples_per_deck=4, neighbors=2,
            )
            self.assertFalse(set(manifest["train_deck_ids"]) & set(manifest["validation_deck_ids"]))
            model = DeckMaskPredictor.load(model_path)
            direct = DeckMaskPredictor.fit([{
                "observed_main": [101], "observed_extra": [103],
                "target_main": {102: 2}, "target_extra": {},
            }], neighbors=2)
            prediction = model.predict([101], [103])
            reloaded = DeckMaskPredictor.load(model_path).predict([101], [103])
            self.assertEqual(prediction, reloaded)
            self.assertIn("main", prediction)
            self.assertIn("extra", prediction)
            self.assertTrue(direct.predict([101], [103])["main"])
            self.assertTrue(Path(str(model_path) + ".metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
