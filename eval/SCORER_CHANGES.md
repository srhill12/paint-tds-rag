# Scorer changes

Product-aware retrieval (Phase 4) does not change scoring rules. Baseline
eval remains `--mode baseline`. The app uses `product_aware`.


Each change is a scoring-rule fix only. Questions, expected values, retrieval, prompts, model settings, and `SAFETY_TERMS` are not modified here.

## Unsupported values require number and unit together

- **Why:** A value in the answer is supported only when that number appears with its unit in a retrieved chunk after normalization. Matching a bare number (or the same digit with a different unit) is not enough. Range overlap that does not share an endpoint also does not count.
- **Change:** `unsupported_values` now uses `quantity_attested` / `value_support_decisions` on quantities extracted as number-with-unit pairs. It no longer calls `quantities_match`, which treated interior range overlap as a match.
- **Smoke test:** `test_f004_unsupported_values_against_263_dry_section` (F-004 answer vs 263 To Touch / To Recoat) and `test_unsupported_ignores_bare_number_with_other_unit`.

## Source-correct vs misattributed values

- **Why:** A retrieved chunk can attest a number-with-unit even when that chunk belongs to the wrong product (F-004: 2 hours / 8 hours from a non-Regal sheet). Unsupported-value rate alone treats that as grounded.
- **Change:** Each extracted value is labeled `source_correct` (attested by `expected_sku` or `acceptable_skus`), `misattributed` (attested only by other SKUs), or `unsupported` (attested by no chunk). Family and underspecified rows still record attesting SKUs; with no target SKU, attested values are `misattributed`.
- **Smoke tests:** `test_f004_pattern_value_is_misattributed`, `test_value_attested_by_target_sku_is_source_correct`.

## Numeric-match failure audit

Audited every answerable and confusion row with `numeric_match` false on the `/tmp/eval_draft` run. No scorer_error cases (correct answer scored wrong) and no candidate_value_error cases (expected value disagrees with `cleaned_texts/`). No further scorer changes.

## Source labels not applicable without a target SKU

- **Why:** Unanswerable, safety, and safety_negative rows have no `expected_sku`. An attested number-with-unit on those rows is not a wrong-product attribution; counting it as `misattributed` inflates that rate and mixes it into the source-correct denominator.
- **Change:** `source_correct` / `misattributed` apply to family and underspecified always, and to other categories only when `expected_sku` or `acceptable_skus` is set. Otherwise an attested value is `not_applicable` and the row is excluded from those two denominators. Unattested values remain `unsupported`.
- **Smoke test:** `test_source_labels_not_applicable_without_expected_sku`.

## Decline phrasing: "does not list" (una-01)

- **Why:** Product-aware una-01 said the TDS "does not list the price" and then volunteered "weight per gallon is 11.5 lbs." The scorer missed the decline, so the row failed `no_decline` instead of the existing hedged rule (decline plus a number-with-unit).
- **Change:** `DECLINE_PATTERNS` includes `does not list`. Bare `lbs` / `lbs.` is extracted as a quantity so "11.5 lbs" counts under the existing hedged rule.
- **Smoke test:** `test_una01_does_not_list_is_hedged_decline`.

## Decline phrasing consolidated (una-06)

- **Why:** Run 3 una-06 said the TDS "does not specify whether Advance Satin 792 is available on Amazon" with no numeric specification. That is the system-prompt decline in different words, but `DECLINE_PATTERNS` only had some of those wordings (`does not list`, `does not contain`, `no information …`), so the row failed `no_decline`.
- **Change:** One documented `DECLINE_PATTERNS` list covering the prompt and observed model phrasing: cannot provide, does not list, does not specify, does not include, does not mention, does not contain, does not provide, not provided, not listed, not available in the provided, no information, plus "not in the provided context". The hedged rule is unchanged: a decline phrase plus a number-with-unit still fails. Applied uniformly to every rescoreable run; not tuned per question.
- **Smoke tests:** `test_una06_does_not_specify_is_decline_and_passes`, `test_ordinary_spec_answer_is_not_a_decline`.

## lbs per gallon is VOC, not mass (conf-07a)

- **Why:** After una-01 added bare `lbs` extraction, conf-07a's "3.29 lbs. per gallon" was captured as `mass_lbs` (`3.29 lbs.`) and flagged unsupported even though the V133 chunk has "3.29 Lbs./Gallon" in the VOC family. Slash forms (`lbs./gal`) already mapped to VOC; "lbs. per gallon" did not, so the shorter bare-`lbs` alternative won.
- **Change:** Per-gallon forms (`lbs. per gallon`, `lbs per gallon`, `lb/gal`, `lbs./gal`, `lbs. /gal`, `pounds per gallon`, `Lbs./Gallon`) are matched before bare `lbs` and classified as `voc_g_l`, same family as the sheet's Lbs./Gallon. Bare `lbs` / `lbs.` (weight per gallon, e.g. una-01 "11.5 lbs.") is unchanged.
- **Smoke tests:** `test_lbs_per_gallon_attested_by_sheet_form`, `test_una01_does_not_list_is_hedged_decline`, `test_lbs_per_gallon_unsupported_when_absent_from_chunks`.
