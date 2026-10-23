#!/usr/bin/env python3
"""Verify every gold.json evidence_span really appears (verbatim, whitespace-tolerant) in the
named source PDF and on the stated page. Graders re-run this check in code - run it before you submit.

Usage: python tools/check_gold_spans.py [--gold data/gold.json] [--papers-dir papers]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import verify as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold.json")
    ap.add_argument("--papers-dir", default="papers")
    a = ap.parse_args()
    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    cache, bad = V.PdfCache(), 0
    for g in gold:
        src = g.get("evidence_source")
        path = V.resolve_source(src, [a.papers_dir, ".", "data"]) if src else None
        if path is None:
            print(f"{g['claim_id']}: SOURCE NOT FOUND ({src})")
            bad += 1
            continue
        pages = cache.get(path)
        pg, mode = V.locate_span(g["evidence_span"], pages, g.get("evidence_page"))
        if pg is None:
            print(f"{g['claim_id']}: SPAN NOT FOUND in {path.name} -> copy it from the PDF exactly")
            bad += 1
        elif pg["page"] != g.get("evidence_page"):
            print(f"{g['claim_id']}: span found ({mode}) but on page {pg['page']}, gold says {g.get('evidence_page')}")
            bad += 1
        else:
            print(f"{g['claim_id']}: OK ({mode}, page {pg['page']}{', OCR' if pg['ocr'] else ''})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
