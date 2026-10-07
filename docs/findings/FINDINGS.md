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
