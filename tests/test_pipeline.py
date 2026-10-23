#!/usr/bin/env python3
"""Offline tests: synthetic papers + a scripted fake LLM. Run: python tests/test_pipeline.py

These tests do NOT use the real papers (they are not in the repo) - they use short synthetic
texts built from the facts in the assignment guide so every code path (text PDF, image-only PDF
+ OCR, verbatim check, figure maths, fail-closed paths) is exercised without network/API keys.
"""
import json
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import verify as V  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    PASS += bool(cond)
    FAIL += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name)


# ---------------------------------------------------------------- synthetic PDFs
def text_pdf(path, pages):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=A4)
    for txt in pages:
        y = 800
        for line in textwrap.wrap(txt, 90):
            c.drawString(40, y, line)
            y -= 14
        c.showPage()
    c.save()


def image_pdf(path, pages):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    imgs = []
    for txt in pages:
        im = Image.new("RGB", (1600, 1100), "white")
        d = ImageDraw.Draw(im)
        y = 40
        for line in textwrap.wrap(txt, 80):
            d.text((40, y), line, fill="black", font=font)
            y += 34
        imgs.append(im)
    imgs[0].save(str(path), save_all=True, append_images=imgs[1:], resolution=150)


FILLER = "Methods. Participants were randomised and followed per protocol at 26 weeks. " * 3
SOLIMIX = [
    "Abstract. In adults with type 2 diabetes suboptimally controlled on basal insulin plus oral drugs, "
    "baseline HbA1c 8.6% was reduced by 1.3% with iGlarLixi and 1.1% with BIAsp 30 at Week 26. "
    "Difference -0.2%, 95% CI -0.4, -0.1, P < 0.001. " + FILLER,
    FILLER,
    "Table 2. Participants who reached HbA1c <7% 42.2 (n=187) 31.8 (n=141) OR 1.65 (1.25, 2.19) in the "
    "iGlarLixi versus BIAsp 30 comparison. " + FILLER,
    "Figure 3. Overall hypoglycemia OR 0.62 (0.47, 0.81). Level 2 hypoglycemia (<54 mg/dL) incidence "
    "6.3% (n=28) OR 0.45 (0.28, 0.73) for iGlarLixi versus BIAsp 30 in adults with type 2 diabetes "
    "suboptimally controlled on basal insulin plus oral drugs. " + FILLER,
]
LIXI_O = [
    FILLER,
    "Results. In adults with type 2 diabetes inadequately controlled on oral agents, changes from baseline "
    "to Week 30 in HbA1c were -1.63% for iGlarLixi, -1.34% for iGlar, P < 0.0001. " + FILLER,
]
LIXI_L_IMG = [
    "Results. Mean body weight decreased by 0.7 kg with iGlarLixi and increased by 0.7 kg with iGlar "
    "(1.4 kg difference, P < 0.0001) in adults with type 2 diabetes insufficiently controlled on basal "
    "insulin. Conclusions: a beneficial effect on body weight."
]
LBL_IMG = [
    "SOLIQUA patient story cover page with no claims to verify here at all.",
    "SOLIQUA is indicated for the treatment of adults patients with Obesity with Insufficiently "
    "controlled type 2 diabetes mellitus to improve glycemic control as an adjunct to diet and "
    "exercise in addition to metformin, when control has not been provided by metformin alone, "
    "or with basal insulin, or with a GLP-1 receptor agonist.",
]

CLAIMS = [
    {"claim_id": "C1", "claim_text": "SOLIQUA is indicated for the treatment of adults with type 2 diabetes "
     "insufficiently controlled on basal insulin or oral agents.", "source": "LBL.pdf"},
    {"claim_id": "C2", "claim_text": "Superior HbA1c reduction with once-daily iGlarLixi (1.3 ± 0.1%) vs "
     "twice-daily premix insulin BIAsp 30 (1.1 ± 0.1%); 95% CI -0.4, -0.1; p < 0.001.",
     "source": "SoliMix_Rosenstock2021.pdf"},
    {"claim_id": "C3", "claim_text": "With SOLIQUA, ~2x greater HbA1c reduction vs basal insulin (iGlar 100 U/mL), "
     "p < 0.0001, from baseline to Week 30.", "source": "LixiLan-O_Rosenstock2016.pdf"},
    {"claim_id": "C4", "claim_text": "More patients achieved HbA1c < 7% with iGlarLixi than premix BIAsp 30: "
     "42.20% vs 31.80% (OR 1.65; 95% CI 1.25-2.19).", "source": "SoliMix_Rosenstock2021.pdf"},
    {"claim_id": "C5", "claim_text": "Weight neutral.", "source": "LixiLan-L_Aroda2016_abstract.pdf"},
    {"claim_id": "C6", "claim_text": "Lower risk of level-2 hypoglycemia (ADA: < 54 mg/dL) with iGlarLixi vs premix "
     "BIAsp 30 [OR 0.45 (0.28, 0.73)].", "source": "SoliMix_Rosenstock2021.pdf"},
    {"claim_id": "C8", "claim_text": "SOLIQUA reduces cardiovascular mortality by 30%.",
     "source": "SoliMix_Rosenstock2021.pdf"},
]


def ext(span, page, t=None, c=None, pop=None, drug=None, comp=None, tp=None, stated=True, extra=None, absent=None,
        found=True):
    return json.dumps({"found": found, "metric": "m", "unit": None, "timepoint": tp, "treatment_value": t,
                       "comparator_value": c, "population_text": pop, "drug_text": drug, "comparator_text": comp,
                       "evidence_span": span, "evidence_page": page, "additional_spans": extra or [],
                       "stated_in_source": stated, "claim_wording_not_in_source": absent or []})


POP_B = "adults with type 2 diabetes suboptimally controlled on basal insulin plus oral drugs"


def scripted(overrides=None):
    """A well-behaved fake reader (what a good LLM would return) - with optional bad behaviour."""
    answers = {
        "C1": ext("SOLIQUA is indicated for the treatment of adults patients with Obesity with Insufficiently "
                  "controlled type 2 diabetes mellitus to improve glycemic control as an adjunct to diet and "
                  "exercise in addition to metformin", 2,
                  pop="adults patients with Obesity with Insufficiently controlled type 2 diabetes mellitus",
                  drug="SOLIQUA", stated=False),
        "C2": ext("was reduced by 1.3% with iGlarLixi and 1.1% with BIAsp 30", 1, t=1.3, c=1.1, pop=POP_B,
                  drug="iGlarLixi", comp="BIAsp 30",
                  extra=[{"page": 1, "span": "Difference -0.2%, 95% CI -0.4, -0.1"}]),
        "C3": ext("changes from baseline to Week 30 in HbA1c were -1.63% for iGlarLixi, -1.34% for iGlar", 2,
                  t=1.63, c=1.34, pop="adults with type 2 diabetes inadequately controlled on oral agents",
                  drug="iGlarLixi", comp="iGlar", tp="Week 30", stated=False),
        "C4": ext("Participants who reached HbA1c <7% 42.2 (n=187) 31.8 (n=141) OR 1.65 (1.25, 2.19)", 3,
                  t=42.2, c=31.8, pop=POP_B, drug="iGlarLixi", comp="BIAsp 30"),
        "C5": ext("Mean body weight decreased by 0.7 kg with iGlarLixi and increased by 0.7 kg with iGlar", 1,
                  t=0.7, c=0.7, pop="insufficiently controlled on basal insulin", drug="iGlarLixi", comp="iGlar",
                  stated=False, absent=["Weight neutral"]),
        "C6": ext("Level 2 hypoglycemia (<54 mg/dL) incidence 6.3% (n=28) OR 0.45 (0.28, 0.73) for iGlarLixi "
                  "versus BIAsp 30", 4, t=0.45, c=None, pop=POP_B, drug="iGlarLixi", comp="BIAsp 30"),
    }
    answers.update(overrides or {})

    def llm(prompt, cid):
        return answers.get(cid, ext("x", 1, found=False))
    return llm


def run_pipeline(tmp, claims, llm):
    papers = Path(tmp) / "papers"
    papers.mkdir(exist_ok=True)
    text_pdf(papers / "SoliMix_Rosenstock2021.pdf", SOLIMIX)
    text_pdf(papers / "LixiLan-O_Rosenstock2016.pdf", LIXI_O)
    image_pdf(papers / "LixiLan-L_Aroda2016_abstract.pdf", LIXI_L_IMG)
    image_pdf(papers / "SOLIQUA_LBL.pdf", LBL_IMG)
    cj = Path(tmp) / "claims.json"
    cj.write_text(json.dumps(claims))
    V.run(str(cj), out_dir=str(Path(tmp) / "outputs"), papers_dir=str(papers), llm=llm)
    return {p["claim_id"]: p for p in json.loads((Path(tmp) / "outputs" / "predictions.json").read_text())}


def main():
    # ---------------- unit tests: arithmetic and parsing live in Python
    check("claim_numbers drops p-values/weeks/doses",
          [v for v, _ in V.claim_numbers("1.3% at Week 30, p < 0.0001, 100 U/mL, 95% CI -0.4, -0.1")] == [1.3, 0.4, 0.1])
    check("multiplier parsed", V.parse_multiplier("~2x greater")[0] == 2.0)
    check("rounding tolerance 1.3 vs 1.34", V.numbers_match((1.3, 1), (1.34, 2)))
    check("rounding tolerance rejects 1.3 vs 1.6", not V.numbers_match((1.3, 1), (1.6, 1)))
    st, comp, _ = V.figure_check("~2x greater reduction", ["were -1.63% for iGlarLixi, -1.34% for iGlar"], 1.63, 1.34)
    check("C3 ratio computed in code -> mismatch", st == "mismatch" and comp["result"] == "1.22x")
    st, comp, _ = V.figure_check("~1.2x greater reduction", ["were -1.63% for iGlarLixi, -1.34% for iGlar"], 1.63, 1.34)
    check("1.2x claim matches 1.22x", st == "match")
    st, _, _ = V.figure_check("~2x", ["were -1.63% and -1.34%"], 2.0, 1.0)
    check("LLM-supplied values not printed in span are rejected (unclear)", st == "unclear")
    check("decide: mismatch beats all", V.decide({"figure": "mismatch", "population": "match",
                                                  "drug_indication": "match", "actually_stated": "match"}) == "overstated")
    check("decide: unclear -> cannot_verify", V.decide({"figure": "match", "population": "unclear",
                                                        "drug_indication": "match", "actually_stated": "match"}) == "cannot_verify")
    check("parse_extraction rejects garbage", V.parse_extraction("not json") is None)
    check("parse_extraction rejects missing keys", V.parse_extraction('{"found": true}') is None)

    # ---------------- OCR fallback detects image-only PDFs
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "img.pdf"
        image_pdf(p, LIXI_L_IMG)
        pages = V.load_pdf(p)
        check("image-only page is OCR'd", pages[0]["ocr"] and "decreased by 0.7 kg" in V.norm(pages[0]["text"]))

    # ---------------- end-to-end with a well-behaved reader
    with tempfile.TemporaryDirectory() as tmp:
        P = run_pipeline(tmp, CLAIMS, scripted())
        check("C1 overstated (dropped qualifiers)", P["C1"]["verdict"] == "overstated")
        check("C2 supported", P["C2"]["verdict"] == "supported")
        check("C3 overstated with computed 1.22x", P["C3"]["verdict"] == "overstated"
              and P["C3"]["computed"]["result"] == "1.22x")
        check("C4 supported (two-column table path)", P["C4"]["verdict"] == "supported")
        check("C5 overstated via OCR'd abstract ('neutral' absent)", P["C5"]["verdict"] == "overstated")
        check("C6 supported", P["C6"]["verdict"] == "supported")
        check("C8 (source silent) -> cannot_verify", P["C8"]["verdict"] == "cannot_verify")
        check("evidence_page comes from Python-verified location", P["C3"]["evidence_page"] == 2)
        check("shape has all brief fields", all(k in P["C3"] for k in
              ("claim_id", "verdict", "checks", "evidence_page", "evidence_span", "computed", "reason")))

    # ---------------- adversarial reader: Python must fail closed / catch it
    with tempfile.TemporaryDirectory() as tmp:
        bad = scripted({
            # extractor grabs the OVERALL hypoglycemia OR 0.62 for C6 (the guide's named trap)
            "C6": ext("Overall hypoglycemia OR 0.62 (0.47, 0.81)", 4, t=0.62, pop=POP_B, drug="iGlarLixi", comp="BIAsp 30"),
            # LLM invents a quote for C2
            "C2": ext("iGlarLixi was dramatically superior to BIAsp 30 in every endpoint", 1, t=1.3, c=1.1,
                      pop=POP_B, drug="iGlarLixi"),
            # LLM 'computes' a flattering 2x itself and hands back made-up arm values for C3
            "C3": ext("changes from baseline to Week 30 in HbA1c were -1.63% for iGlarLixi, -1.34% for iGlar", 2,
                      t=3.2, c=1.6, pop="inadequately controlled on oral agents", drug="iGlarLixi", comp="iGlar"),
            # LLM returns invalid JSON for C4
            "C4": "Sure! Here is my answer: it is supported.",
        })
        P = run_pipeline(tmp, CLAIMS, bad)
        check("C6 with 0.62 span is NOT supported", P["C6"]["verdict"] != "supported")
        check("invented quote -> cannot_verify", P["C2"]["verdict"] == "cannot_verify")
        check("fabricated arm values never give supported", P["C3"]["verdict"] != "supported")
        check("invalid JSON -> cannot_verify", P["C4"]["verdict"] == "cannot_verify")

    # ---------------- crash in the LLM layer fails closed per claim
    with tempfile.TemporaryDirectory() as tmp:
        def boom(prompt, cid):
            raise RuntimeError("network down")
        P = run_pipeline(tmp, CLAIMS[:2], boom)
        check("LLM exception -> cannot_verify, no traceback", all(p["verdict"] == "cannot_verify" for p in P.values()))

    # ---------------- no API key: clean message, all cannot_verify
    with tempfile.TemporaryDirectory() as tmp:
        import os
        for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
            os.environ.pop(k, None)
        cj = Path(tmp) / "claims.json"
        cj.write_text(json.dumps(CLAIMS))
        V.run(str(cj), out_dir=str(Path(tmp) / "o"), papers_dir=str(Path(tmp)))
        P = json.loads((Path(tmp) / "o" / "predictions.json").read_text())
        check("no key -> every claim cannot_verify", len(P) == len(CLAIMS) and all(p["verdict"] == "cannot_verify" for p in P))

    # ---------------- unresolvable source
    with tempfile.TemporaryDirectory() as tmp:
        P = run_pipeline(tmp, [{"claim_id": "C9", "claim_text": "x 1.0%", "source": "nonexistent.pdf"}], scripted())
        check("missing source PDF -> cannot_verify", P["C9"]["verdict"] == "cannot_verify")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
