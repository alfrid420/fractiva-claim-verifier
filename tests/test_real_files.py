#!/usr/bin/env python3
"""Run the REAL pipeline (real PDFs, OCR, real claims.json, retrieval, verbatim checks, Python
judge) with a scripted *reader* standing in for the LLM.

The scripted reader quotes passages sliced out of the real extracted page text (so every span is
genuinely verbatim) and labels numbers the way a careful LLM should. Python then decides.
This validates everything EXCEPT the LLM itself. Run: python tests/test_real_files.py
Skips cleanly if the real PDFs are not in papers/.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import verify as V  # noqa: E402

PAPERS = ROOT / "papers"
NEEDED = ["SOLIQUA_LBL.pdf", "SoliMix_Rosenstock2021.pdf", "LixiLan-O_Rosenstock2016.pdf",
          "LixiLan-L_Aroda2016_abstract.pdf", "Tabak2020_PSM.pdf"]
PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    PASS += bool(cond)
    FAIL += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra and not cond else ""))


def main():
    if not all((PAPERS / n).is_file() for n in NEEDED) or not (ROOT / "data/claims.json").is_file():
        print("real PDFs / claims.json not present - skipping")
        return 0
    cache = V.PdfCache()
    claims = {c["claim_id"]: c for c in V.load_claims(ROOT / "data/claims.json")}

    def page(fname, n, variant="text"):
        return cache.get(PAPERS / fname)[n - 1][variant]

    def cut(fname, n, pattern, variant="text"):
        """Slice a verbatim span out of the real page text with a regex."""
        m = re.search(pattern, page(fname, n, variant), re.S)
        assert m, f"pattern not found on {fname} p{n}: {pattern}"
        return m.group(0)

    SM, LO, LL, TB, LBL = ("SoliMix_Rosenstock2021.pdf", "LixiLan-O_Rosenstock2016.pdf",
                           "LixiLan-L_Aroda2016_abstract.pdf", "Tabak2020_PSM.pdf", "SOLIQUA_LBL.pdf")
    POP_SM = cut(SM, 1, r"type 2 diabetes\s+suboptimally controlled on basal insulin plus oral\s+antihyperglycemic drugs")

    def ext(span, pg, **kw):
        d = {"found": True, "metric": "m", "unit": None, "timepoint": None, "treatment_value": None,
             "comparator_value": None, "population_text": None, "drug_text": None, "comparator_text": None,
             "evidence_span": span, "evidence_page": pg, "additional_spans": [], "stated_in_source": True,
             "claim_wording_not_in_source": []}
        d.update(kw)
        return json.dumps(d)

    answers = {
        "C1": ext(cut(LBL, 4, r"is indicated for the treatment of adults patients with Obesity.{0,200}?in addition to metformin"), 4,
                  population_text=cut(LBL, 4, r"adults patients with Obesity with insufficiently controlled type 2 diabetes mellitus"),
                  drug_text="SOLIQUA", stated_in_source=False),
        "C2": ext(cut(SM, 1, r"was reduced by 1\.3% with iGlarLixi and 1\.1% with BIAsp 30"), 1,
                  treatment_value=1.3, comparator_value=1.1, population_text=POP_SM,
                  drug_text="once-daily iGlarLixi", comparator_text="twice-daily BIAsp 30",
                  additional_spans=[{"page": 5, "span": "-1.3 ± 0.1 %"}, {"page": 5, "span": "-1.1 ± 0.1 %"},
                                    {"page": 5, "span": cut(SM, 5, r"Superiority:.{0,120}?<0.001\n")}]),
        "C3": ext(cut(LO, 13, r"changes from baseline to Week 30 in HbA1c were −1\.63% for iGlarLixi,\s+−1\.34% for\s+iGlar"), 13,
                  treatment_value=1.63, comparator_value=1.34, timepoint="Week 30",
                  population_text=cut(LO, 3, r"type 2 diabetes inadequately\s+controlled on metformin ± a second oral glucose-lowering drug"),
                  drug_text="iGlarLixi", comparator_text="iGlar", stated_in_source=False),
        "C4": ext("42.2 (n=187) 31.8 (n=141) 1.65 (1.25, 2.19)", 6,
                  treatment_value=42.2, comparator_value=31.8, population_text=POP_SM,
                  drug_text="iGlarLixi", comparator_text="BIAsp 30"),
        "C5": ext(cut(LL, 2, r"Mean body weight decreased by 0\.7 kg with iGlarLixi and increased by\s+0\.7 kg with iGlar"), 2,
                  treatment_value=0.7, comparator_value=0.7,
                  population_text=cut(LL, 2, r"inadequately controlled, basal insulin-treated, long-standing type 2"),
                  drug_text="iGlarLixi", comparator_text="iGlar", stated_in_source=False,
                  claim_wording_not_in_source=["Weight neutral"]),
        "C6": ext("6.3 (n=28) 12.9 (n=57) 55% 0.45 (0.28, 0.73)", 7,
                  treatment_value=6.3, comparator_value=12.9, population_text=POP_SM,
                  drug_text="iGlarLixi", comparator_text="BIAsp 30"),
        "C7": ext(cut(TB, 5, r"weight\s+decreased with iGlarLixi by an LS mean \(SE\) of\s+0\.62 kg \(0\.24\)"), 5,
                  treatment_value=0.62, comparator_value=0.70,
                  population_text="patients with type 2 diabetes uncontrolled on basal insulin",
                  drug_text="iGlarLixi", comparator_text="basal bolus"),
    }

    def run(overrides=None):
        a = dict(answers)
        a.update(overrides or {})
        out = {}
        for cid, c in claims.items():
            out[cid] = V.verify_claim(c, [str(PAPERS), str(ROOT / "data")], cache, lambda p, i, a=a: a[i])
        return out

    P = run()
    for cid, p in P.items():
        print(f"  {cid}: {p['verdict']:<14} p{p['evidence_page']}  {p['checks']}\n       {p['reason'][:170]}")
    check("C1 overstated (obesity/diet+exercise/metformin dropped)", P["C1"]["verdict"] == "overstated")
    check("C2 supported", P["C2"]["verdict"] == "supported", P["C2"]["reason"])
    check("C3 overstated, ratio computed 1.22x", P["C3"]["verdict"] == "overstated" and P["C3"]["computed"]["result"] == "1.22x", P["C3"]["reason"])
    check("C4 supported (Figure 2 row pairing)", P["C4"]["verdict"] == "supported", P["C4"]["reason"])
    check("C5 overstated ('neutral' absent from OCR'd abstract)", P["C5"]["verdict"] == "overstated", P["C5"]["reason"])
    check("C6 supported", P["C6"]["verdict"] == "supported", P["C6"]["reason"])
    check("C7 never 'supported' (3-source synthesis -> human review)", P["C7"]["verdict"] != "supported", P["C7"]["reason"])

    # Adversarial: the extractor grabs the overall-hypoglycaemia OR 0.62 for C6
    bad = run({"C6": ext(cut(SM, 6, r"0\.62 \[95% CI 0\.47, 0\.81\]"), 6, treatment_value=0.62,
                         population_text=POP_SM, drug_text="iGlarLixi", comparator_text="BIAsp 30")})
    check("C6 given the 0.62 overall-hypoglycemia OR is NOT supported", bad["C6"]["verdict"] != "supported", bad["C6"]["reason"])
    # Adversarial: a quote that is not in the source
    bad = run({"C4": ext("iGlarLixi was dramatically superior in every endpoint", 6, treatment_value=42.2, comparator_value=31.8)})
    check("invented quote -> cannot_verify", bad["C4"]["verdict"] == "cannot_verify")
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
