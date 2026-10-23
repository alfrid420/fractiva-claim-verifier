# WRITEUP — Fractiva claim verifier (GoNuts.ai research-intern assignment)

## 1. Approach
Seven stages: load -> extract text per page -> OCR fallback for image-only pages -> keyword retrieval of
candidate pages -> LLM reads and returns strict JSON (every call logged) -> Python runs four checks (figure,
population, drug/indication, actually-stated) -> verdict from a fixed logic table. The LLM never computes or
judges: it finds and quotes; Python verifies the quote exists verbatim on the page, recomputes ratios, and
decides. Fail-closed because a confidently wrong "supported" is the costliest error; every uncertainty
becomes `cannot_verify` with a reason.

## 2. Results
Final run: `gemini-3.5-flash-lite`, Tesseract OCR on, output-token limit raised (see section 5).

```
PER-CLAIM
claim  gold           predicted      result
C1     overstated     overstated     ok
C2     supported      supported      ok
C3     overstated     overstated     ok
C4     supported      supported      ok
C5     overstated     overstated     ok
C6     supported      supported      ok
C7     overstated     cannot_verify  miss

ACCURACY: 6/7 = 85.7%

CONFUSION MATRIX (rows = gold, columns = predicted)
                supported      overstated     cannot_verify
supported       3              0              0
overstated      0              3              1
cannot_verify   0              0              0

FALSE SUPPORTED (predicted supported, gold is not): 0
```

## 3. The hardest table
Two extraction problems dominated. (1) The SOLIQUA label (LBL) and the LixiLan-L abstract are image-only
PDFs with no text layer: a plain extractor returns almost nothing, so those pages scored < 100 characters
and the OCR fallback (PyMuPDF render + Tesseract) is the only reason C1 and C5 could be verified at all.
(2) SoliMix's results live in figures, not clean tables — the HbA1c <7% numbers for C4 (42.2% vs 31.8%,
OR 1.65) and the level-2 hypoglycemia numbers for C6 (OR 0.45) sit in Figure 2 / Figure 3 rows whose text
layer interleaves two columns. The pipeline stores a second, row-reconstructed view of digit-dense pages
(words grouped by vertical position) so a number stays attached to its row label; the verbatim-span check
then confirms the pairing before any number is used.

## 4. One real failure
C7 (gold: overstated; predicted: cannot_verify). C7 stitches one claim out of three sources (LixiLan-L,
SoliMix, and a propensity-matched analysis). Every check my pipeline could run came back clean — the weight
numbers reconcile in all three sources (-0.7 vs +0.7 kg; -0.7 vs +1.2 kg; -0.62 vs +0.70 kg) — but the
actually-stated check returned `unclear`: "claim combines 3 cited sources; no single source states it, so
it needs human review." The overstatement in C7 lives in details no single check catches: the LBL prints
p<0.0001 under a panel whose paper reports only P<0.001, and the third panel is a post-hoc matched analysis,
not a trial. The pipeline's refusal is the designed behaviour for stitched claims, but against gold it is
still a miss, and I am reporting it as one.

## 5. Where AI misled me
Four real instances, all visible in my run logs:
- The AI-generated code defaulted to model `gemini-2.5-flash`, which Google had already retired for new
  API keys. Every call returned HTTP 404, the program failed closed, and my first eval showed 0/7 — it looked
  like the pipeline was worthless when in fact it had never made a single successful call. Fix: override
  the model via the GEMINI_MODEL environment variable.
- The AI set the output limit to 1500 tokens. The current Gemini models spend tokens "thinking" out of the
  same budget, so replies were cut off mid-JSON (logged lengths 138–157 characters, ending mid-word) and
  failed validation. Raising the limit to 8000 ended the truncation; the failures after that were quota
  problems, not parsing problems (next bullet).
- The AI's setup instructions assumed Tesseract was installed. It was not, so the image-only PDFs silently
  produced no text and C1 first came back "source does not address this claim" — a correct-looking verdict
  built on missing evidence.
- On the free tier, repeated runs exhausted the per-model quota (HTTP 429, plus 503s). The fix was
  operational, not code-deep: switch to a model with a fresh quota pool and lengthen the retry backoff
  from seconds to 30s steps, so one rate-limited claim no longer poisoned a run.

## 6. What not to trust
- OCR'd spans (LBL page-4 small print) should be re-checked by eye before being quoted as evidence.
- C7 rests partly on a post-hoc propensity-matched analysis; three designs stitched into one claim.
- C2's CI exists at two levels (97.5% non-inferiority / 95% superiority) with the same bounds.
- `claim_wording_not_in_source` is nominated by the LLM; Python only confirms the word is absent from the
  source, so a paraphrase could be wrongly flagged (errs toward `overstated`, never toward `supported`).
- A "0/7" or all-`cannot_verify` result from this pipeline usually means the run never happened (key,
  model, OCR or quota) — check `predictions.json` reasons and `llm_log.jsonl` before believing it.
- Free-tier model names and quotas change without notice; a run is only reproducible with the model name
  recorded in the LLM log.

## 7. Next step
Give verify.py a per-claim resume: store each completed claim's prediction and, on re-run, only call the
LLM for claims that failed with a transient error (429/503), instead of re-rolling every claim. Measure it
by re-running eval.py after a deliberately interrupted run — accuracy should never go down when a run is
resumed — and by testing on a small self-made set of unseen claims, which is also how the live interview
round will use it.
