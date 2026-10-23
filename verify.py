#!/usr/bin/env python3
"""Fractiva claim verifier.

Usage:  python verify.py data/claims.json

Design: the LLM is a READER (finds + quotes text, labels numbers as JSON).
Python is the JUDGE (all arithmetic, all comparisons, the verdict).
Anything that goes wrong -> `cannot_verify` with a reason. Never a guess.

Stages: 1 load -> 2 extract text -> 3 OCR fallback -> 4 retrieve pages
        -> 5 LLM extraction (logged) -> 6 Python checks -> 7 verdict + write
"""
import argparse
import datetime
import difflib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------
# Named constants (all thresholds live here - nothing is buried in the logic)
# ----------------------------------------------------------------------------
MIN_PAGE_CHARS = 100            # below this a page is treated as image-only -> OCR
OCR_DPI = 200
OCR_DPI_RETRY = 300             # second attempt for tiny print (e.g. prescribing info)
TOP_PAGES_PER_SOURCE = 6        # max candidate pages kept per cited PDF
PROMPT_CHAR_BUDGET = 40000      # total page text sent per claim (split across cited PDFs)
NUMERIC_PAGE_BONUS = 3          # tables/figures (digit-dense pages) are where results live
RESULT_CUE_BONUS = 3            # claims are about results: favour pages that report results
MAX_PAGE_CHARS_IN_PROMPT = 7000
ROUNDING_HALF = 0.5             # tolerance = 0.5 * last printed digit
MULTIPLIER_TOLERANCE = 0.15     # '~2x' vs computed ratio
SPAN_TOKEN_COVERAGE = 0.95      # order-tolerant fallback for interleaved columns
MIN_SPAN_TOKENS_FOR_TOLERANT = 6
SNIPPET_COVERAGE = 0.80         # population/drug snippets must really be on the pages
FUZZY_WORD_RATIO = 0.80         # OCR-garble tolerance when asserting a word is absent
MIN_ABSENT_WORD_LEN = 5
MAX_EXTRA_SPANS = 4
IMAGE_PAGE_TEXT_CHARS = 1500    # a page with an image and less text than this is ALSO OCR'd
ROW_GAP_PT = 9                  # words closer than this vertically share one reconstructed row
NUMERIC_PAGE_DIGIT_RATIO = 0.05 # pages denser than this get the row-reconstructed view in the prompt
PAGE_HINT_BOOST = 6
LABEL_WINDOW_CHARS = 900
MULTI_SOURCE_MIN = 2            # claims citing >= this many sources are syntheses -> human review
PATH_FUZZY_CUTOFF = 0.60

LLM_TEMPERATURE = 0
LLM_MAX_OUTPUT_TOKENS = 8000
LLM_MAX_RETRIES = 3
LLM_TIMEOUT_S = 90
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

CHECK_NAMES = ("figure", "population", "drug_indication", "actually_stated")
VERDICTS = ("supported", "overstated", "cannot_verify")

ENDPOINT_TERMS = {
    "hba1c", "weight", "hypoglycemia", "hypoglycaemia", "glycemic", "mortality",
    "bmi", "fpg", "ldl", "nausea", "vomiting", "incidence", "odds", "hazard",
}
STOPWORDS = set("""a an and are as at be been by for from has have in into is it its of on or
that the their this to was were which with within than then also vs versus not no more most
less greater lower higher over under after before between among per both each other such
patients patient adults adult study trial treatment group groups using used use""".split())

# ----------------------------------------------------------------------------
# Controlled vocabularies for the Python checks (regexes run on normalised text)
# ----------------------------------------------------------------------------
DRUG_VOCAB = {
    "iglarlixi": r"iglarlixi|soliqua|glargine[^.;]{0,25}lixisenatide|lixisenatide",
    "iglar": r"\biglar\b|insulin glargine(?![^.;]{0,40}lixi)",
    "biasp30": r"biasp|premix|pre-mix|biphasic insulin aspart",
    "basal_bolus": r"basal[- ]bolus",
    "metformin_drug": r"\bmetformin\b",
}
POP_CONDITION = {
    "t2dm": r"type 2 diabetes|t2dm|\bt2d\b",
    "t1dm": r"type 1 diabetes|t1dm",
}
POP_BACKGROUND = {
    "basal_insulin": r"basal insulin",
    "oral_agents": r"oral (?:glucose[- ]lowering|antidiabetic|anti-diabetic|agents?|drugs?|medications?|antihyperglyc\w+)|\boads?\b",
    "metformin": r"\bmetformin\b",
    "sglt2": r"sglt-?2",
    "glp1": r"glp-?1",
}
# A background-therapy word in the claim only counts as a *population* statement when it sits
# in a population-style phrase ("controlled on ...", "switching from ..."), not as a comparator.
POP_CUE = r"(?:controlled|uncontrolled|inadequately|insufficiently|suboptimally|treated|switch\w*|failing|patients|adults)\b[^.;()]{0,70}"
POP_RESTRICTIVE = {
    "obesity": r"obes",
    "pregnancy": r"pregnan",
    "pediatric": r"pediatric|paediatric|children|adolescent",
    "renal": r"renal|dialysis|kidney|\bckd\b",
    "elderly": r"elderly|older adults|aged\s*(?:>=|≥)\s*65",
}
INDICATION_QUALIFIERS = {
    "diet_exercise": r"diet and exercise",
    "metformin_addon": r"in addition to metformin|with metformin",
}
LABEL_MARKER = r"\bis indicated\b|\bindicated for\b|\bindication\b"

NOISE_PATTERNS = [   # numbers that are NOT the figures being claimed
    r"\b[pP]\s*[<=>≤≥]\s*\d*\.?\d+(?:e-?\d+)?",
    r"[<>≤≥]\s*\d+(?:\.\d+)?",                      # thresholds: 'HbA1c < 7%', '< 54 mg/dL'
    r"\b(?:biasp|iglar|glargine|insulin)\s*\d+(?:\s*u/ml)?",   # drug-name numbers: 'BIAsp 30', 'iGlar 100'
    r"\bhba1c\b",
    r"\blevel[- ]?\d\b",                                # 'level-2 hypoglycemia'
    r"\btype[- ]?\d\b",                                 # 'type 2 diabetes'
    r"\bglp-?1\b",
    r"\b\d+(?:\.\d+)?\s*%\s*CI\b",
    r"\bweeks?\s*\d+\b",
    r"\b\d+\s*-?\s*(?:weeks?|wks?)\b",
    r"\b\d+(?:\.\d+)?\s*(?:mg/dl|mg|u/ml|units?|u\b|mmol/l|mcg|µg)",
    r"\bfigure\s*\d+|\btable\s*\d+|\bpage\s*\d+",
]
MULT_RE = re.compile(
    r"(?:~|≈|approx\.?\s*|about\s*)?\b(\d+(?:\.\d+)?)\s*(?:x\b|×|-?\s*fold\b|\s+times\b)", re.I)
MULT_WORDS = {"twice": 2.0, "double": 2.0, "doubled": 2.0, "two-fold": 2.0, "twofold": 2.0,
              "triple": 3.0, "tripled": 3.0, "three-fold": 3.0}
NUM_RE = re.compile(r"(?<![\w.])[-−–]?\d+(?:\.\d+)?")


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
def norm(s):
    """Whitespace/unicode-normalised lowercase text (for verbatim + vocab checks)."""
    s = unicodedata.normalize("NFKC", s or "").replace("\x01", "-").replace("\ufffe", "")  # font-glyph minus / soft hyphen
    s = (s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
          .replace("\u00ad", "").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"'))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return re.sub(r"hba ?1 ?c", "hba1c", s)     # PDF subscripts split 'HbA' / '1c'


def parse_numbers(text):
    """-> list of (abs_value, decimals) for every number printed in `text`."""
    out = []
    for m in NUM_RE.finditer(norm(text)):
        raw = m.group(0).lstrip("-−–")
        dec = len(raw.split(".")[1]) if "." in raw else 0
        out.append((float(raw), dec))
    return out


def numbers_match(a, b):
    """Tolerance = half of the last printed digit of the LESS precise number."""
    (va, da), (vb, db) = a, b
    return abs(va - vb) <= ROUNDING_HALF * 10 ** (-min(da, db)) + 1e-9


def parse_multiplier(claim_text):
    m = MULT_RE.search(claim_text)
    if m:
        return float(m.group(1)), m.span()
    low = norm(claim_text)
    for w, v in MULT_WORDS.items():     # 'twice-daily' / 'double-blind' are not multipliers
        mm = re.search(r"\b" + re.escape(w) + r"\b(?![- ]?(?:daily|a day|per day|weekly|blind|dummy|masked))", low)
        if mm:
            return v, None
    return None, None


def claim_numbers(claim_text):
    """Numbers the claim asserts (p-values, CI level, dose, week numbers removed)."""
    text = claim_text
    _, span = parse_multiplier(text)
    if span:
        text = text[:span[0]] + " " + text[span[1]:]
    for pat in NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.I)
    return parse_numbers(text)


def claim_timepoints(text):
    low = norm(text)
    pts = set(int(x) for x in re.findall(r"\bweeks?\s*(\d+)\b", low))
    pts |= set(int(x) for x in re.findall(r"\b(\d+)\s*-?\s*(?:weeks?|wks?)\b", low))
    return pts


def terms_in(text, vocab):
    low = norm(text)
    return {k for k, pat in vocab.items() if re.search(pat, low)}


# ----------------------------------------------------------------------------
# STAGE 2 + 3: extraction with OCR fallback
# ----------------------------------------------------------------------------
def _ocr_image(render, dpi_list=(OCR_DPI, OCR_DPI_RETRY)):
    import pytesseract
    text = ""
    for dpi in dpi_list:
        text = pytesseract.image_to_string(render(dpi))
        if len(text.strip()) >= MIN_PAGE_CHARS:
            break
    return text


def rows_from_words(words, gap=ROW_GAP_PT):
    """Rebuild table/figure rows from word coordinates: words whose vertical centres chain within
    `gap` points share one line. Journal figures print a row label BETWEEN its two value lines, so
    plain text extraction scrambles them; this keeps label + values + OR together."""
    if not words:
        return ""
    ws = sorted(words, key=lambda w: (w["top"] + w["bottom"]) / 2)
    groups, cur = [], [ws[0]]
    for w in ws[1:]:
        if (w["top"] + w["bottom"]) / 2 - (cur[-1]["top"] + cur[-1]["bottom"]) / 2 <= gap:
            cur.append(w)
        else:
            groups.append(cur)
            cur = [w]
    groups.append(cur)
    return "\n".join(" ".join(w["text"] for w in sorted(g, key=lambda w: (round(w["x0"] / 60), w["top"], w["x0"])))
                     for g in groups)


def _iter_pages(path):
    """Yield (page_no, text, text_sorted, words, n_images, render(dpi)->PIL image)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
    if fitz is not None:
        from PIL import Image
        doc = fitz.open(str(path))
        for i, p in enumerate(doc, 1):
            words = [{"text": w[4], "x0": w[0], "top": w[1], "bottom": w[3]} for w in p.get_text("words")]

            def render(dpi, p=p):
                pix = p.get_pixmap(dpi=dpi)
                return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            yield i, p.get_text("text"), p.get_text("text", sort=True), words, len(p.get_images(full=True)), render
        return
    try:  # fallback used only where PyMuPDF is unavailable (e.g. offline test sandbox)
        import pdfplumber
        import pypdfium2 as pdfium
    except ImportError:
        raise RuntimeError("PyMuPDF is not installed (pip install -r requirements.txt)")
    pdf = pdfium.PdfDocument(str(path))
    with pdfplumber.open(str(path)) as plumb:       # pdfplumber supplies word boxes + image counts only
        for i, pg in enumerate(plumb.pages, 1):
            text = pdf[i - 1].get_textpage().get_text_range().replace("\r\n", "\n")

            def render(dpi, i=i):
                return pdf[i - 1].render(scale=dpi / 72).to_pil().convert("RGB")
            yield i, text, text, pg.extract_words(), len(pg.images), render


def _clean_text(t):
    """Drop invisible soft-hyphen markers some PDFs embed inside words ('dia\ufffebetes')."""
    return (t or "").replace("\ufffe", "").replace("\u00ad", "")


def _finish_page(num, text, text_sorted, words, n_images, render):
    text, text_sorted = _clean_text(text), _clean_text(text_sorted)
    rec = {"page": num, "text": text, "text_sorted": text_sorted, "text_rows": rows_from_words(words),
           "ocr": False, "n_images": n_images, "chars": len(text.strip()), "ocr_error": None}
    # OCR when the page is (nearly) empty OR carries an image plus little text: a scanned abstract can
    # still have a stray text-layer footer, so a bare character-count rule alone would skip it.
    if rec["chars"] < MIN_PAGE_CHARS or (n_images and rec["chars"] < IMAGE_PAGE_TEXT_CHARS):
        try:
            ocr_text = _ocr_image(render)
            for k in ("text", "text_sorted", "text_rows"):
                rec[k] = (rec[k] + "\n" + ocr_text) if rec[k].strip() else ocr_text
            rec["ocr"] = True
            rec["chars"] = len(rec["text"].strip())
        except Exception as e:                  # tesseract missing, etc.
            rec["ocr_error"] = f"{type(e).__name__}: {e}"
    return rec


def load_pdf(path):
    """-> [{page, text, text_sorted, text_rows, ocr, n_images, chars, ocr_error}] (1-based pages).
    No LLM in here."""
    return [_finish_page(*args) for args in _iter_pages(path)]


# ----------------------------------------------------------------------------
# STAGE 1 helpers: claims + source resolution
ID_KEYS = ("claim_id", "id", "claimId")
TEXT_KEYS = ("claim_text", "text", "claim", "statement", "claim_statement")
SRC_KEYS = ("cited_paper", "source_pdf", "source", "cited_source", "paper", "source_file", "sources",
            "cited_sources", "pdf", "papers", "source_pdfs")
POP_KEYS = ("population", "claimed_population")
MOL_KEYS = ("molecule", "drug", "intervention")


def _first(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None


def load_claims(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for k in ("claims", "items", "data"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:
            data = [dict(v, claim_id=k) if isinstance(v, dict) else {"claim_id": k, "claim_text": v}
                    for k, v in data.items()]
    claims = []
    for i, c in enumerate(data, 1):
        src = _first(c, SRC_KEYS)
        if isinstance(src, (str, dict)):
            src = [src]
        names = []
        for s_ in src or []:
            if isinstance(s_, dict):
                s_ = _first(s_, ("file", "path", "name", "pdf", "source"))
            if s_:
                names.append(str(s_))
        claims.append({"claim_id": str(_first(c, ID_KEYS) or f"C{i}"), "claim_text": str(_first(c, TEXT_KEYS) or ""),
                       "population": _first(c, POP_KEYS), "molecule": _first(c, MOL_KEYS),
                       "sources": names, "raw": c})
    return claims


def _alnum(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _stem(p):
    return _alnum(Path(str(p)).stem)


def list_pdfs(search_dirs):
    """{filename: path} over the search dirs (first directory wins on duplicates)."""
    out = {}
    for d in search_dirs:
        d = Path(d)
        if d.is_dir():
            for f in sorted(d.glob("*.pdf")):
                out.setdefault(f.name, f)
    return out


def resolve_source(name, search_dirs):
    """Exact path -> same file name in search dirs -> fuzzy stem match. None if unresolved."""
    p = Path(name)
    if p.is_file():
        return p
    pdfs = list_pdfs(search_dirs)
    if p.name in pdfs:
        return pdfs[p.name]
    want = _stem(name)
    for f in pdfs.values():
        if _stem(f) == want:
            return f
    for f in pdfs.values():
        if want and (want in _stem(f) or _stem(f) in want):
            return f
    best = difflib.get_close_matches(want, [_stem(f) for f in pdfs.values()], n=1, cutoff=PATH_FUZZY_CUTOFF)
    return next((f for f in pdfs.values() if best and _stem(f) == best[0]), None)


def resolve_sources(names, search_dirs):
    """Cited-source strings -> (paths, unresolved, page_hint).  A string may hold a file name
    ('papers/X.pdf (Prescribing Information, page 4)') and/or descriptive mentions
    ('... plus LixiLan-L abstract and SoliMix'); each paper is matched by file name or by the
    distinctive first word of its file name ('solimix', 'lixilanl', 'tabak2020')."""
    pdfs = list_pdfs(search_dirs)
    found, unresolved, hint = [], [], None
    for raw in names:
        hits = []
        for fn in re.findall(r"[\w\-./]+\.pdf", raw, flags=re.I):
            p = resolve_source(fn, search_dirs)
            if p:
                hits.append(p)
        cited = _alnum(raw)
        for name, p in pdfs.items():
            key = _alnum(Path(name).stem.split("_")[0])
            if len(key) >= 5 and key in cited:
                hits.append(p)
        if not hits:
            unresolved.append(raw)
            continue
        m = re.search(r"\bpage\s*(\d+)", raw, re.I)
        if m and hint is None:
            hint = (hits[0].resolve(), int(m.group(1)))
        for p in hits:
            if all(p.resolve() != q.resolve() for q in found):   # same file via relative/absolute path
                found.append(p)
    return found, unresolved, hint


# ----------------------------------------------------------------------------
# STAGE 4: retrieval (cheap keyword scoring; keeps page boundaries)
# ----------------------------------------------------------------------------
def _number_variants(raw):
    v = {raw}
    if "." in raw:
        v.add(raw.rstrip("0").rstrip("."))
    return v


RESULT_CUES = re.compile(r"\bls mean\b|least squares|from baseline|primary (?:efficacy )?end ?point|\bresults\b|"
                         r"odds ratio|\bci\b|confidence interval|\bp ?[<=]", re.I)


def retrieve(claim_text, pages, top_k=TOP_PAGES_PER_SOURCE, hint_page=None, char_budget=PROMPT_CHAR_BUDGET):
    low = norm(claim_text)
    nums = {s for m in NUM_RE.finditer(low) for s in _number_variants(m.group(0).lstrip("-−–"))}
    words = {w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", low) if w not in STOPWORDS}
    scored = []
    for pg in pages:
        t = norm(pg["text"])
        score = sum(3 for n in nums if re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", t))
        score += sum((2 if w in ENDPOINT_TERMS else 1) for w in words if w in t)
        if score > 0 and RESULT_CUES.search(t):
            score += RESULT_CUE_BONUS
        if score > 0 and _is_numeric_page(pg["text"]):
            score += NUMERIC_PAGE_BONUS
        if hint_page is not None and pg["page"] == hint_page:
            score += PAGE_HINT_BOOST
        if score > 0:
            scored.append((score, pg))
    scored.sort(key=lambda x: (-x[0], x[1]["page"]))
    chosen, used = [], 0
    for _, pg in scored:
        size = min(len(pg["text"]), MAX_PAGE_CHARS_IN_PROMPT)
        if len(chosen) >= top_k or (chosen and used + size > char_budget):
            break
        chosen.append(pg)
        used += size
    return chosen


# ----------------------------------------------------------------------------
# STAGE 5: LLM extraction (strict JSON, every call logged)
# ----------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are a careful READER of clinical-trial papers and drug labels.
You do NOT judge whether the claim is right and you NEVER calculate anything (no ratios,
differences or percentages). Your only job is to find the passage(s) of the SOURCE that speak
to the CLAIM and report what is printed there, as JSON.

CLAIM (marketing text to be checked):
\"\"\"{claim}\"\"\"
{meta}
SOURCE PAGES (text extracted from the cited PDF; may contain OCR errors or interleaved
two-column table text). Numeric pages are shown twice: the normal text, and a ROW-RECONSTRUCTED
view in which words at the same vertical position share a line - use it to see which row label a
number belongs to:
{pages}

Return ONLY one JSON object, no prose, no markdown fences, with exactly these keys:
{{
  "found": true|false,                 // false if no page above addresses the claim's endpoint/topic
  "metric": "endpoint the claim is about, as named in the source",
  "unit": "unit as printed, or null",
  "timepoint": "e.g. 'Week 30' / '26 weeks' as printed, or null",
  "treatment_value": number|null,      // the source's printed value for the claimed (treatment) arm
  "comparator_value": number|null,     // the source's printed value for the comparator arm
  "population_text": "verbatim phrase describing who was enrolled / who the label is for, or null",
  "drug_text": "verbatim phrase naming the molecule/dose/regimen of the treatment arm, or null",
  "comparator_text": "verbatim phrase naming the comparator, or null",
  "evidence_span": "ONE sentence or table row copied CHARACTER-FOR-CHARACTER from a page above",
  "evidence_page": <page number the evidence_span is on>,
  "additional_spans": [{{"page": <n>, "span": "verbatim text"}}],   // up to 4 more verbatim passages
  "stated_in_source": true|false,      // true only if the source itself STATES what the claim says
  "claim_wording_not_in_source": ["short phrases copied from the CLAIM whose meaning the source never states"]
}}

Rules:
1. Every span must be copied exactly from the pages above. Never fix typos, never paraphrase,
   never merge text from two places into one span. If you cannot quote it, set found=false.
2. treatment_value / comparator_value must be numbers PRINTED in your evidence span, for the
   endpoint the claim is about. If the source reports several similar endpoints (e.g. overall vs
   level-2 hypoglycaemia; an incidence odds ratio vs an event-rate ratio; HbA1c change vs HbA1c
   <7% responders) choose the one that matches the claim and name it in "metric". In a table or
   figure, confirm the row label of every number you use (the same number can appear in two rows).
3. Include in evidence_span / additional_spans the passages that contain the numbers the claim
   states (point estimates, confidence intervals, arm values), so they can be checked.
4. stated_in_source = false when the source only supports a weaker, different or opposite
   statement than the claim (list the over-reaching phrases in claim_wording_not_in_source);
   leave that list empty when the claim's wording is genuinely stated.
5. If unsure, prefer found=false / null over guessing.
"""

REQUIRED_KEYS = ("found", "metric", "unit", "timepoint", "treatment_value", "comparator_value",
                 "population_text", "drug_text", "comparator_text", "evidence_span",
                 "evidence_page", "stated_in_source")


def _is_numeric_page(text):
    t = text or ""
    return bool(t) and sum(c.isdigit() for c in t) / len(t) >= NUMERIC_PAGE_DIGIT_RATIO


def build_prompt(claim_text, candidates, population=None, molecule=None):
    blocks = []
    for pg in candidates:
        t = pg["text"].strip()
        if len(t) > MAX_PAGE_CHARS_IN_PROMPT:
            t = t[:MAX_PAGE_CHARS_IN_PROMPT] + "\n[...page truncated...]"
        tag = " (OCR)" if pg["ocr"] else ""
        block = f"=== FILE: {pg['file']} | PAGE {pg['page']}{tag} ===\n{t}"
        rows = pg.get("text_rows", "").strip()
        if _is_numeric_page(t) and rows and rows != t:
            block += (f"\n--- same page, ROW-RECONSTRUCTED VIEW ---\n"
                      f"{rows[:MAX_PAGE_CHARS_IN_PROMPT]}")
        blocks.append(block)
    meta = ""
    if population or molecule:
        meta = ("CLAIM METADATA (from the claims file - context only, NOT evidence): "
                f"population={population!r}; molecule={molecule!r}\n")
    return EXTRACTION_PROMPT.format(claim=claim_text.strip(), meta=meta, pages="\n\n".join(blocks))


def parse_extraction(raw):
    """Validate the LLM JSON. Returns dict or None (None => fail closed)."""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        d = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except Exception:
            d = None
    if not isinstance(d, dict) or any(k not in d for k in REQUIRED_KEYS):
        return None
    if not isinstance(d["found"], bool) or not isinstance(d["stated_in_source"], bool):
        return None
    d.setdefault("additional_spans", [])
    d.setdefault("claim_wording_not_in_source", [])
    if not isinstance(d["additional_spans"], list) or not isinstance(d["claim_wording_not_in_source"], list):
        return None
    return d


def get_provider():
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def make_llm(log_path):
    """Returns llm(prompt, claim_id) -> raw text. Appends every call to the JSONL log."""
    provider = get_provider()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(rec):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def llm(prompt, claim_id):
        if provider == "gemini":
            model, url = GEMINI_MODEL, f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            params = {"temperature": LLM_TEMPERATURE, "maxOutputTokens": LLM_MAX_OUTPUT_TOKENS,
                      "responseMimeType": "application/json"}
            body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": params}
            headers = {"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]}
        else:
            model, url = ANTHROPIC_MODEL, "https://api.anthropic.com/v1/messages"
            params = {"temperature": LLM_TEMPERATURE, "max_tokens": LLM_MAX_OUTPUT_TOKENS}
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], **params}
            headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                       "x-api-key": os.environ["ANTHROPIC_API_KEY"]}
        raw, err = None, None
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as r:
                    data = json.loads(r.read().decode())
                raw = (data["candidates"][0]["content"]["parts"][0]["text"] if provider == "gemini"
                       else data["content"][0]["text"])
                err = None
                break
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            time.sleep(30 * attempt)
        _log({"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), "claim_id": claim_id,
              "provider": provider, "model": model, "parameters": params, "prompt": prompt,
              "response": raw, "error": err})
        if raw is None:
            raise RuntimeError(f"LLM call failed ({err})")
        return raw
    return llm


# ----------------------------------------------------------------------------
# STAGE 6: Python checks (no LLM maths, ever)
# ----------------------------------------------------------------------------
def _variants(pg):
    return (pg["text"], pg["text_sorted"], pg.get("text_rows", ""))


def locate_span(span, pages, preferred_page=None):
    """Find `span` in the source pages. Returns (page_dict, mode) or (None, None).
    exact = whitespace-normalised substring; tolerant = >=95% of tokens present (two-column
    tables interleave under extraction, so order can differ)."""
    n = norm(span)
    if not n:
        return None, None
    ordered = sorted(pages, key=lambda p: 0 if p["page"] == preferred_page else 1)
    for pg in ordered:
        for variant in _variants(pg):
            if n in norm(variant):
                return pg, "exact"
    toks = n.split()
    if len(toks) >= MIN_SPAN_TOKENS_FOR_TOLERANT:
        need = Counter(toks)
        for pg in ordered:
            for variant in _variants(pg):
                have = Counter(norm(variant).split())
                cov = sum(min(c, have[t]) for t, c in need.items()) / len(toks)
                if cov >= SPAN_TOKEN_COVERAGE:
                    return pg, "tolerant"
    return None, None


def snippet_supported(snippet, pages):
    """Is an LLM-reported snippet really on the pages (token coverage)?"""
    toks = norm(snippet).split()
    if not toks:
        return False
    for pg in pages:
        have = set().union(*(set(norm(v).split()) for v in _variants(pg)))
        if sum(t in have for t in toks) / len(toks) >= SNIPPET_COVERAGE:
            return True
    return False


def _as_float(v):
    try:
        return abs(float(v))
    except (TypeError, ValueError):
        return None


def figure_check(claim_text, span_texts, t_val, c_val):
    """-> (status|None, computed|None, detail). None = claim states no checkable figure."""
    mult, _ = parse_multiplier(claim_text)
    cnums = claim_numbers(claim_text)
    if mult is None and not cnums:
        return None, None, "claim states no checkable figure"
    span_nums = [n for s in span_texts for n in parse_numbers(s)]

    def validated(v):
        """LLM-reported value counts only if it is literally printed in a verified span."""
        f = _as_float(v)
        return f if f is not None and any(abs(f - sn[0]) < 1e-6 for sn in span_nums) else None
    t, c = validated(t_val), validated(c_val)
    statuses, details, computed = [], [], None

    if mult is not None:
        if t is None or c in (None, 0):
            statuses.append("unclear")
            details.append("multiplier claim but source arm values are not both printed in the quoted text")
        else:
            ratio = t / c                                        # computed HERE, in code
            ok = abs(ratio - mult) <= MULTIPLIER_TOLERANCE
            statuses.append("match" if ok else "mismatch")
            computed = {"claimed": f"~{mult:g}x", "from_source": [t, c], "result": f"{ratio:.2f}x"}
            details.append(f"claimed ~{mult:g}x but source values {t:g} / {c:g} give {ratio:.2f}x")

    if cnums:
        matched = [n for n in cnums if any(numbers_match(n, s) for s in span_nums)]
        unmatched = [n for n in cnums if n not in matched]
        prim = [v for v in (t, c) if v is not None]
        unmatched_prim = [v for v in prim if not any(numbers_match((v, 2), n) or numbers_match(n, (v, 2))
                                                     for n in cnums)]
        if not unmatched:
            st = "match"
        elif unmatched_prim and (not matched or len(unmatched) == len(unmatched_prim)):
            st = "mismatch"
        else:
            st = "unclear"
        statuses.append(st)
        details.append(f"claimed numbers {[v for v, _ in cnums]}: "
                       f"{'all found' if st == 'match' else 'not found in quoted source text: ' + str([v for v, _ in unmatched])}")
        if computed is None:
            computed = {"claimed": ", ".join(f"{v:g}" for v, _ in cnums),
                        "from_source": [sn[0] for sn in span_nums
                                        if any(numbers_match(n, sn) for n in cnums)],
                        "result": "all claimed numbers reconcile" if st == "match"
                                  else "claimed numbers do not reconcile"}
    status = ("mismatch" if "mismatch" in statuses else "unclear" if "unclear" in statuses else "match")
    return status, computed, "; ".join(details)


def _claim_population_terms(claim_text):
    low = norm(claim_text)
    terms = terms_in(low, POP_CONDITION) | terms_in(low, POP_RESTRICTIVE)
    for k, pat in POP_BACKGROUND.items():
        if re.search(POP_CUE + r"(?:" + pat + ")", low) or re.search(r"(?:" + pat + r")[^.;()]{0,25}(?:uncontrolled|failure)", low):
            terms.add(k)
    return terms


def _population_vocab_terms(text):
    return terms_in(text, POP_CONDITION) | terms_in(text, POP_BACKGROUND) | terms_in(text, POP_RESTRICTIVE)


def population_check(claim_ctx, population_text, span_texts, pages):
    if not population_text or not snippet_supported(population_text, pages):
        return "unclear", "source population text not found on the quoted pages"
    src = norm(population_text + " " + " ".join(span_texts))
    page_terms = _population_vocab_terms(norm(" ".join(pg["text"] for pg in pages)))
    claim_terms = _claim_population_terms(claim_ctx)
    extra = claim_terms - _population_vocab_terms(src) - page_terms
    if extra:
        return "mismatch", f"claim names population {sorted(extra)} that the source does not state"
    dropped = terms_in(src, POP_RESTRICTIVE) - terms_in(claim_ctx, POP_RESTRICTIVE)
    if dropped:
        return "mismatch", f"source restricts to {sorted(dropped)} but the claim omits it (broader than source)"
    return "match", f"claim population consistent with source ({population_text[:80]!r})"


def drug_check(claim_ctx, drug_text, comparator_text, timepoint, span_texts, pages):
    src = norm(" ".join([drug_text or "", comparator_text or ""] + list(span_texts)))
    if not src.strip():
        return "unclear", "no source drug/comparator text"
    claim_drugs, src_drugs = terms_in(claim_ctx, DRUG_VOCAB), terms_in(src, DRUG_VOCAB)
    issues, unclear = [], []
    missing = claim_drugs - src_drugs
    if missing:   # the quoted text may omit a name that the page itself states
        page_drugs = terms_in(norm(" ".join(pg["text"] for pg in pages)), DRUG_VOCAB)
        missing -= page_drugs
    if missing:
        issues.append(f"claim names {sorted(missing)} which the source text does not")
    claim_tp = claim_timepoints(claim_ctx)
    if claim_tp:
        src_tp = claim_timepoints(" ".join([timepoint or ""] + list(span_texts)))
        if not src_tp:
            unclear.append(f"claim time point {sorted(claim_tp)} wk not confirmed in quoted source")
        elif not claim_tp <= src_tp:
            issues.append(f"time point differs: claim {sorted(claim_tp)} wk vs source {sorted(src_tp)} wk")
    if re.search(LABEL_MARKER, src):   # label wording: dropped qualifiers widen the indication
        label = src
        for pg in pages:
            t = norm(pg["text"])
            m = re.search(LABEL_MARKER, t)
            if m:
                label += " " + t[m.start(): m.start() + LABEL_WINDOW_CHARS]
        dropped = terms_in(label, INDICATION_QUALIFIERS) - terms_in(claim_ctx, INDICATION_QUALIFIERS)
        if dropped:
            issues.append(f"label qualifiers dropped from claim: {sorted(dropped)}")
    if issues:
        return "mismatch", "; ".join(issues)
    if unclear or (not claim_drugs and not src_drugs):
        return "unclear", "; ".join(unclear) or "no drug/comparator terms could be compared"
    return "match", "molecule/comparator/time point consistent"


def absent_words(phrases, claim_text, all_pages):
    """Words from LLM-nominated claim phrases that appear NOWHERE in the cited documents
    (OCR-garble tolerant). The phrase must really be in the claim."""
    vocab = set()
    for pg in all_pages:
        vocab |= set(re.findall(r"[a-z]+", norm(pg["text"]) + " " + norm(pg["text_sorted"])))
    vocab_list = sorted(vocab)
    claim_low, out = norm(claim_text), []
    for ph in phrases:
        if not isinstance(ph, str) or norm(ph) not in claim_low:
            continue
        for w in re.findall(r"[a-z]+", norm(ph)):
            if len(w) < MIN_ABSENT_WORD_LEN or w in STOPWORDS:
                continue
            if any(t.startswith(w[:5]) for t in vocab) or difflib.get_close_matches(w, vocab_list, n=1, cutoff=FUZZY_WORD_RATIO):
                continue
            out.append(w)
    return sorted(set(out))


def decide(checks):
    vals = [v for v in checks.values() if v is not None]
    if "mismatch" in vals:
        return "overstated"
    if (checks["actually_stated"] == "match" and checks["population"] == "match"
            and checks["drug_indication"] == "match" and checks["figure"] in ("match", None)):
        return "supported"
    return "cannot_verify"


def make_prediction(cid, verdict, checks, page, span, computed, reason, source=None):
    return {"claim_id": cid, "verdict": verdict, "checks": checks, "evidence_page": page,
            "evidence_span": span, "computed": computed, "reason": reason, "evidence_source": source}


def cannot_verify(cid, reason, checks=None):
    return make_prediction(cid, "cannot_verify",
                           checks or {k: "unclear" for k in CHECK_NAMES}, None, None, None, reason)


def _clean_snippet(snippet, pages):
    """Keep an LLM-reported descriptive snippet only if it really is on the candidate pages."""
    return snippet if snippet and snippet_supported(snippet, pages) else ""


def evaluate(claim_id, claim_text, ext, candidates, all_pages, population=None, molecule=None, n_sources=1):
    """Pure function: LLM extraction + page text -> prediction. No network, no LLM."""
    prim_page, mode = locate_span(ext["evidence_span"] or "", candidates, _as_int(ext["evidence_page"]))
    if prim_page is None:
        return cannot_verify(claim_id, "Quoted evidence span was not found verbatim in the source; "
                                       "refusing to rely on an unverifiable quote.")
    verified = [(prim_page, ext["evidence_span"])]
    for extra in ext["additional_spans"][:MAX_EXTRA_SPANS]:
        if isinstance(extra, dict) and isinstance(extra.get("span"), str):
            pg, _ = locate_span(extra["span"], candidates, _as_int(extra.get("page")))
            if pg is not None:
                verified.append((pg, extra["span"]))
    span_texts = [s for _, s in verified]
    pop_ctx = f"{claim_text} {population or ''}"
    drug_ctx = f"{claim_text} {molecule or ''}"
    pop_text = _clean_snippet(ext["population_text"], candidates)
    drug_text = _clean_snippet(ext["drug_text"], candidates)
    comp_text = _clean_snippet(ext["comparator_text"], candidates)

    fig, computed, fig_detail = figure_check(claim_text, span_texts, ext["treatment_value"], ext["comparator_value"])
    pop, pop_detail = population_check(pop_ctx, pop_text, span_texts, candidates)
    drug, drug_detail = drug_check(drug_ctx, drug_text, comp_text, ext["timepoint"], span_texts, candidates)

    absent = absent_words(ext["claim_wording_not_in_source"], claim_text, all_pages)
    if absent:
        stated, stated_detail = "mismatch", f"claim wording {absent} appears nowhere in the cited source"
    elif n_sources >= MULTI_SOURCE_MIN:
        stated, stated_detail = "unclear", (f"claim combines {n_sources} cited sources; no single source states it, "
                                            "so it needs human review")
    elif ext["stated_in_source"]:
        stated, stated_detail = "match", f"quoted verbatim ({mode} match)"
    else:
        stated, stated_detail = "unclear", "source does not clearly state what the claim says"

    checks = {"figure": fig, "population": pop, "drug_indication": drug, "actually_stated": stated}
    verdict = decide(checks)
    details = {"figure": fig_detail, "population": pop_detail, "drug_indication": drug_detail,
               "actually_stated": stated_detail}
    bad = [f"{k}={v} ({details[k]})" for k, v in checks.items() if v in ("mismatch", "unclear")]
    reason = ("All applicable checks match the quoted source." if verdict == "supported"
              else "; ".join(bad))
    return make_prediction(claim_id, verdict, checks, prim_page["page"], ext["evidence_span"],
                           computed, reason, prim_page["file"])


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
class PdfCache:
    def __init__(self):
        self.cache = {}

    def get(self, path):
        key = str(Path(path).resolve())
        if key not in self.cache:
            pages = load_pdf(path)
            for p in pages:
                p["file"] = Path(path).name
            self.cache[key] = pages
        return self.cache[key]


def verify_claim(claim, search_dirs, cache, llm):
    cid = claim["claim_id"]
    if not claim["claim_text"].strip():
        return cannot_verify(cid, "Claim text is empty.")
    paths, unresolved, hint = resolve_sources(claim["sources"], search_dirs)
    if not claim["sources"]:   # no cited source in the claims file: search every paper we have
        paths = sorted(set(list_pdfs(search_dirs).values()))
    if not paths:
        return cannot_verify(cid, f"Cited source could not be located: {unresolved or 'none listed'}.")
    all_pages, candidates, notes = [], [], []
    for p in paths:
        pages = cache.get(p)
        all_pages += pages
        blank = [pg["page"] for pg in pages if pg["chars"] < MIN_PAGE_CHARS]
        if blank:
            notes.append(f"{Path(p).name}: pages {blank} unreadable")
        h = hint[1] if hint and hint[0] == Path(p).resolve() else None
        candidates += retrieve(claim["claim_text"], pages, hint_page=h,
                               char_budget=PROMPT_CHAR_BUDGET // max(1, len(paths)))
    if unresolved:
        notes.append(f"unresolved sources {unresolved}")
    if not candidates:
        return cannot_verify(cid, "No page of the cited source mentions the claim's endpoint or numbers. "
                                  + "; ".join(notes))
    raw = llm(build_prompt(claim["claim_text"], candidates, claim.get("population"), claim.get("molecule")), cid)
    ext = parse_extraction(raw)
    if ext is None:
        return cannot_verify(cid, "LLM output was not valid structured JSON; failing closed.")
    if not ext["found"]:
        return cannot_verify(cid, "The cited source does not address this claim (nothing found)."
                                  + (" " + "; ".join(notes) if notes else ""))
    pred = evaluate(cid, claim["claim_text"], ext, candidates, all_pages, claim.get("population"),
                    claim.get("molecule"), n_sources=len(paths) + len(unresolved))
    if notes and pred["verdict"] == "cannot_verify":
        pred["reason"] += " [" + "; ".join(notes) + "]"
    return pred


def run(claims_path, out_dir="outputs", papers_dir="papers", llm=None):
    claims = load_claims(claims_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    claims_dir = Path(claims_path).resolve().parent
    search_dirs = [papers_dir, claims_dir, claims_dir.parent, claims_dir.parent / "papers", ".", Path(__file__).parent,
                   Path(__file__).parent / "papers"]
    preds = []
    if llm is None and get_provider() is None:
        print("ERROR: no LLM API key found. Set GEMINI_API_KEY (or ANTHROPIC_API_KEY) and re-run.\n"
              "Writing cannot_verify for every claim.", file=sys.stderr)
        preds = [cannot_verify(c["claim_id"], "No LLM API key configured (GEMINI_API_KEY); nothing was verified.")
                 for c in claims]
    else:
        llm = llm or make_llm(out_dir / "llm_log.jsonl")
        cache = PdfCache()
        for c in claims:
            try:
                p = verify_claim(c, search_dirs, cache, llm)
            except Exception as e:     # whole-claim guard: any failure fails closed
                p = cannot_verify(c["claim_id"], f"Processing error ({type(e).__name__}: {e}); failing closed.")
            preds.append(p)
            print(f"{p['claim_id']}: {p['verdict']}")
    (out_dir / "predictions.json").write_text(json.dumps(preds, indent=2, ensure_ascii=False), encoding="utf-8")
    return preds


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify marketing claims against cited papers.")
    ap.add_argument("claims", help="path to claims.json")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--papers-dir", default="papers")
    a = ap.parse_args(argv)
    if not Path(a.claims).is_file():
        print(f"ERROR: claims file not found: {a.claims}", file=sys.stderr)
        return 2
    preds = run(a.claims, a.out_dir, a.papers_dir)
    print(f"Wrote {Path(a.out_dir) / 'predictions.json'} ({len(preds)} claims)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
