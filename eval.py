#!/usr/bin/env python3
"""Compare outputs/predictions.json with data/gold.json. Pure Python, no LLM, no network.

Usage: python eval.py [--gold data/gold.json] [--pred outputs/predictions.json]
"""
import argparse
import json
import sys
from pathlib import Path

LABELS = ["supported", "overstated", "cannot_verify"]


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("claims") or data.get("predictions") or list(data.values())
    return {str(r["claim_id"]): r for r in data}


def evaluate(gold, pred):
    rows, matrix = [], {g: {p: 0 for p in LABELS} for g in LABELS}
    missing_pred = sorted(set(gold) - set(pred))
    missing_gold = sorted(set(pred) - set(gold))
    correct = false_supported = 0
    for cid in sorted(set(gold) & set(pred), key=lambda x: (len(x), x)):
        g, p = gold[cid]["verdict"], pred[cid]["verdict"]
        if g not in LABELS or p not in LABELS:
            print(f"WARNING: {cid} has unknown verdict gold={g!r} pred={p!r}", file=sys.stderr)
            continue
        matrix[g][p] += 1
        ok = g == p
        correct += ok
        fs = p == "supported" and g != "supported"
        false_supported += fs
        rows.append((cid, g, p, ok, fs))
    return rows, matrix, correct, false_supported, missing_pred, missing_gold


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold.json")
    ap.add_argument("--pred", default="outputs/predictions.json")
    a = ap.parse_args(argv)
    for f in (a.gold, a.pred):
        if not Path(f).is_file():
            print(f"ERROR: file not found: {f}", file=sys.stderr)
            return 2
    gold, pred = load(a.gold), load(a.pred)
    rows, matrix, correct, fs, miss_p, miss_g = evaluate(gold, pred)
    n = len(rows)

    print("PER-CLAIM")
    print(f"{'claim':<7}{'gold':<15}{'predicted':<15}{'result'}")
    for cid, g, p, ok, f in rows:
        tag = "FALSE SUPPORTED" if f else ("ok" if ok else "miss")
        print(f"{cid:<7}{g:<15}{p:<15}{tag}")
    print()
    acc = f"{correct}/{n} = {correct / n:.1%}" if n else "n/a (no claims joined)"
    print(f"ACCURACY: {acc}")
    print("\nCONFUSION MATRIX (rows = gold, columns = predicted)")
    print(f"{'':<16}" + "".join(f"{p:<15}" for p in LABELS))
    for g in LABELS:
        print(f"{g:<16}" + "".join(f"{matrix[g][p]:<15}" for p in LABELS))
    print(f"\nFALSE SUPPORTED (predicted supported, gold is not): {fs}")
    if miss_p:
        print(f"NOTE: {len(miss_p)} gold claim(s) have no prediction: {', '.join(miss_p)}")
    if miss_g:
        print(f"NOTE: {len(miss_g)} predicted claim(s) have no gold entry: {', '.join(miss_g)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
