# Fractiva claim verifier (GoNuts.ai research-intern assignment)

Checks each marketing claim in `data/claims.json` against the paper it cites and returns
`supported` / `overstated` / `cannot_verify`. **The LLM only reads; Python does every
comparison, every calculation and the verdict. Anything uncertain fails closed to `cannot_verify`.**

## Setup + run (5 commands)

```bash
# 0. system dependency (OCR for the image-only LBL + LixiLan-L abstract):
#    macOS: brew install tesseract   |   Ubuntu/Debian: sudo apt install tesseract-ocr   |   Windows: UB-Mannheim installer
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here        # Windows PowerShell: $env:GEMINI_API_KEY="..."
python verify.py data/claims.json          # -> outputs/predictions.json + outputs/llm_log.jsonl
python eval.py                             # accuracy, confusion matrix, FALSE SUPPORTED count
```

* Key is read from the environment only (`GEMINI_API_KEY`; `ANTHROPIC_API_KEY` also works).
  With no key the program prints a clear message and writes `cannot_verify` for every claim.
* Optional model override: `GEMINI_MODEL` / `ANTHROPIC_MODEL`.
* PDFs must be in `papers/`. `claims.json` fields used: `claim_id`, `text`, `population`, `molecule`, `cited_paper`
  (a file name and/or descriptive mentions such as `plus LixiLan-L abstract and SoliMix`; `page N` boosts that page).
  A claim that cites more than one source is treated as a synthesis and is never auto-`supported`.

## Pipeline (verify.py)

1. **Load** the claims file given on the command line (nothing hard-coded).
2. **Extract** text page-by-page with PyMuPDF (page numbers kept; two text orderings stored because
   two-column tables interleave).
3. **OCR fallback**: a page with < 100 characters is rendered (200 dpi, retry 300) and OCR'd with Tesseract.
4. **Retrieve** the top-3 pages per cited PDF by claim numbers + endpoint/comparator keywords.
5. **LLM extraction**: strict-JSON prompt; output validated in code; every call logged to `outputs/llm_log.jsonl`.
6. **Python checks**: figure (regex numbers, rounding tolerance, ratios computed in code),
   population, drug/indication/time point (controlled vocabularies), actually-stated
   (quote must exist verbatim on the page; wording the LLM says is unstated must be absent from the whole source).
7. **Verdict**: any mismatch -> `overstated`; all match -> `supported`; otherwise `cannot_verify`.

## Other files

* `data/gold.json` - reference verdicts (Part 1). `tools/check_gold_spans.py` checks every gold quote against the PDFs.
* `tests/test_pipeline.py` - offline tests on synthetic PDFs (incl. OCR) with a scripted fake LLM, incl. adversarial LLM behaviour (`pip install reportlab` first).
* `tests/test_real_files.py` - runs the real PDFs + real `claims.json` through retrieval, OCR, verbatim checks and the Python judge with a scripted reader standing in for the LLM (no API key needed).
