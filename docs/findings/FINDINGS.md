# Findings Log

Observed failures and notable behavior, recorded as found. Each finding links to the eval category or fix that addresses it.

## F-001: Unspecified product answered with unrelated products

- Date: 2026-10-07
- Query: "Is this product safe to use in enclosed spaces?"
- Observed: Safety routing triggered correctly and showed the SDS notice. The secondary TDS answer then stated "No, this product should only be used outdoors or in a well-ventilated area," drawing on three industrial epoxies (HP4000, HP4600, HP4100) the user never named.
- Expected: Ask the user which product they mean before answering.
- Risk: Staff could apply a product-specific answer to a different product.
- Also observed: Some TDS documents (HP series) include precautionary safety text. TDS safety content is partial and varies by sheet; the SDS remains the authoritative source.
- Status: Open. Added to eval as the "underspecified" category. Planned fix in Phase 4: product-reference detection (shared with SKU filtering) triggers a clarifying question when no product is named.
- Evidence: screenshots/F-001-unspecified-product.png

## F-002: SKU query retrieves the wrong product, then volunteers an unrelated value

- Date: 2026-10-07
- Query: "what is the voc content of n549?"
- Observed: N549 (Regal Select Interior Eggshell) is in the corpus, but the top retrieved source was Arborcoat 329/C329. The model correctly said the provided documents did not cover N549, then offered a VOC value of 49 g/L attributed to "Data Sheet 49," apparently matching the digits in the code.
- Expected: Retrieve the N549 sheet and answer from it, or decline without offering values from other products.
- Risk: A skimming reader could take the unrelated 49 g/L as N549's VOC content.
- Contradicts: README Known Limitations claim that SKU codes improve retrieval precision.
- Status: Open. Planned fix in Phase 4: SKU detection with metadata filtering before semantic search. The README claim will be corrected with eval evidence.
- Evidence: screenshots/F-002-sku-retrieval-miss.png

## F-003: Product family name resolved to one family member without asking (pass with note)

- Date: 2026-10-07
- Query: "does regal select give off fumes while drying?"
- Observed: Safety routing triggered on "fumes" and the SDS notice appeared first. The secondary answer declined cleanly, stating the TDS documents do not cover fumes during drying, with no unrelated values offered. Top source was Regal Select Exterior Flat 261.
- Note: "Regal Select" names a family spanning interior (N547 to N551) and exterior (261 to 263, N400 to N403) products. Retrieval picked one member silently. Harmless here because the answer declined, but on a specification question it would present one product's value as the family's.
- Minor UI: the SDS documentation link appears twice (inside the notice and below it).
- Status: Pass for routing and decline. Family ambiguity folded into the Phase 4 product-reference fix: a name matching multiple products triggers a clarifying question.
- Evidence: screenshots/F-003-family-name-ambiguity.png

## F-004: Family-level question answered with values that contradict the top cited sheet

- Date: 2026-10-07
- Query: "what is the drying time of regal select?"
- Observed: Answer stated "Regal Select dries tack-free in 2 hours and can be recoated in 8 hours," cited only as "(Technical Data Sheet)" with no product named. Top retrieved source was Regal Select Exterior Soft Gloss 263.
- Verified against cleaned_texts/263_TDS_US.txt: Dry Time @ 77 F, 50% RH: To Touch 1 hour, To Recoat 4 hours. The answer's values are double the top source's values.
- Expected: Ask which Regal Select product is meant (the name spans 15 interior and exterior products), or answer for a named product with the product identified in the answer.
- Risk: Highest observed so far. A confident specification presented as applying to a whole product family, not matching the top cited sheet, with no product named for the reader to verify.
- Open question: whether the values came from another retrieved sheet (misattribution) or from no retrieved sheet (fabrication). To be resolved by the eval's per-question logs, which record all retrieved SKUs.
- Status: Open. Addressed by Phase 4 product-reference detection (clarify on family names) plus a requirement that answers name the product and SKU each value comes from.
- Evidence: screenshots/F-004-family-spec-mismatch.png

### Addendum to F-003

- cleaned_texts/263_TDS_US.txt line 333 contains ventilation guidance for application and drying. The F-003 decline ("TDS documents do not cover fumes during drying") may reflect a retrieval miss rather than a true absence. Safety routing to the SDS remains the correct primary response.

### Resolution of F-004 open question

- Checked dry and recoat values across all 15 Regal Select sheets in the corpus (261, 262, 263, N400, N401, N403, W103, W096, W105, N547, N548, N549, N550, N551, 551).
- To Touch: 1 hour on 12 sheets, 2 hours on W103, W096, W105.
- To Recoat: 1 to 2 hours (N547 to N550), 3 hours (N551, 551), 4 hours (all exterior sheets).
- No Regal Select sheet states an 8-hour recoat, and none uses the term "tack-free."
- Conclusion: the 8-hour recoat value is not supported by any Regal Select document. It was either fabricated or drawn from a non-Regal sheet among the other retrieved sources without attribution. The eval's per-question logs will distinguish these cases.
- Also demonstrates that a family-level question has no single correct answer: recoat times within the family range from 1 to 4 hours.

## F-005: Paraphrased safety question not routed; answered with an unrelated product's hazard warning

- Date: 2026-10-07
- Query: "will the smell bother my newborn?"
- Observed: Safety routing did not trigger ("smell" and "newborn" are not in SAFETY_TERMS), so no SDS notice appeared. The answer began "According to the Technical Data Sheet for [Product Name - Not Specified]" and then quoted a birth defect hazard warning from Ultra Spec HP Acrylic Metal Primer HP04, an industrial metal primer the user never mentioned.
- Expected: Route to the SDS notice, and ask which product is meant before giving any product-specific content.
- Risk: High. A parent could be alarmed by an unrelated product's reproductive hazard warning, or, for a different retrieval, falsely reassured. The model recognized the product was unspecified and answered anyway.
- Grounding note: The quoted text is accurate to the HP04 sheet. This is misapplication, not fabrication.
- Confirms: Governance Decision 6 limitation that keyword routing misses paraphrased safety questions.
- Status: Open. Router terms deliberately not tuned before the baseline. Paraphrased safety questions will be added to the eval; any term-list expansion will be reported as a measured change. Product-reference detection (Phase 4) addresses the unspecified product.
- Evidence: screenshots/F-005-paraphrased-safety-miss.png
