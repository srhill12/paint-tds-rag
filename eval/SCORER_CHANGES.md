# Scorer changes

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
