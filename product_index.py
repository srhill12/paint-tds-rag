"""Product catalog, family mapping, and question-side product resolution."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from config import FAMILIES_PATH, MANIFEST_PATH

KIND_SINGLE = "single"
KIND_FAMILY = "family"
KIND_NONE = "none"

SPEC_TERMS: tuple[str, ...] = (
    "dry time",
    "drying time",
    "application temperature",
    "spread rate",
    "film thickness",
    "flash point",
    "square feet",
    "sq. ft",
    "sq ft",
    "recoat",
    "coverage",
    "voc",
    "sheen",
    "gloss",
    "solids",
    "viscosity",
    "thinning",
    "dry",
)

DEICTIC_PHRASES: tuple[str, ...] = (
    "this product",
    "that product",
    "the product",
    "this paint",
    "that paint",
    "the paint",
)

# Longest first. "ben" is matched as a whole word only.
KNOWN_FAMILIES: tuple[str, ...] = tuple(
    sorted(
        (
            "Regal Select",
            "Aura Bath & Spa",
            "Aura",
            "ben",
            "Super Hide Zero VOC",
            "Super Hide",
            "Corotech",
            "Advance",
            "Ultra Spec HP",
            "Ultra Spec 500",
            "Ultra Spec EXT",
            "Ultra Spec Ext",
            "Ultra Spec Masonry",
            "Ultra Spec Hi-Build",
            "Ultra Spec",
            "Super Spec HP",
            "Super Spec",
            "High Performance",
            "Fresh Start",
            "Arborcoat",
            "Woodluxe",
            "Scuff-X",
            "Eco Spec WB",
            "Eco Spec",
            "Element Guard",
            "Stays Clear",
            "Satin Impervo",
            "Coronado",
            "Cryli Cote",
            "Elastite",
            "Super Kote 5000",
            "Super Kote 3000",
            "Super Kote",
            "Tough Walls",
            "Tru-Flex",
            "Tuffcrete",
            "Texcrete",
            "INSL-X",
            "INSLX",
            "Muresco",
            "GarageGuard",
            "Garageguard",
            "Cabinet Coat",
            "Stix",
            "Sure Step",
            "Waterblock",
            "Lead Block",
            "Blockout",
            "Aqua Lock",
            "Color-Changing Ceiling Paint",
            "Kitchen & Bath",
            "Floor & Patio",
            "Latex Floor & Patio",
            "Benjamin Moore",
            "Fresh Start",
            "Prime All",
            "Prime Lock Plus",
            "Max Block",
            "Hottrax",
            "Fire Retardant Paint",
            "FinalTouch",
            "Freezer-Kote",
            "ClearCoat PRO",
            "All-Purpose Citrus Cleaner",
            "Athletic Field Marking Paste",
            "Chlorinated Rubber Swimming Pool Paint",
            "Decorative Alkyd Aerosol",
            "Rust Preventative Alkyd Aerosol",
            "Rust-A-Void",
            "Waterborne Pool Paint",
            "Rubber Based Pool Paint",
            "Epoxy Pool Coating",
            "Waterborne Ceiling Paint",
            "Drywall Primer",
            "Multi-Purpose Primer",
            "High-Build Exterior Texture",
            "Acrylic Fast-Set Traffic Paint",
            "Alkyd Traffic Paint",
            "Latex Traffic Paint",
            "Latex Field Marking Paint",
            "MULTAPPLY",
            "Marvelux",
            "Calcimine Recoater",
            "Seal Lock",
            "Stripple",
            "Tough Shield",
            "Acrylic Knockdown",
            "High Build Peel Bonding Primer",
        ),
        key=len,
        reverse=True,
    )
)

_DESCRIPTOR_TOKENS = {
    "interior",
    "exterior",
    "paint",
    "primer",
    "finish",
    "flat",
    "matte",
    "eggshell",
    "satin",
    "pearl",
    "gloss",
    "semi",
    "semi-gloss",
    "waterborne",
    "alkyd",
    "acrylic",
    "latex",
    "enamel",
    "stain",
    "sealer",
    "high",
    "low",
    "soft",
    "premium",
    "and",
    "plus",
    "the",
    "for",
    "with",
}

_SKU_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{0,8}\d[A-Za-z0-9.\-]*)(?![A-Za-z0-9])"
)
_TRAILING_SKU_RE = re.compile(
    r"\s+[A-Z]{0,8}\d[A-Za-z0-9.\-/]*(?:\s*\*.*)?$",
    flags=re.IGNORECASE,
)


def _strip_marks(text: str) -> str:
    cleaned = (text or "").replace("®", " ").replace("™", " ").replace("©", " ")
    cleaned = cleaned.replace("–", "-").replace("—", "-").replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_name(text: str) -> str:
    cleaned = _strip_marks(text).lower()
    cleaned = re.sub(r"[^a-z0-9&+\-]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _compile_phrase(phrase: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<!\w){re.escape(phrase)}(?!\w)",
        flags=re.IGNORECASE,
    )


_SPEC_PATTERNS = tuple(
    (term, _compile_phrase(term)) for term in SPEC_TERMS
)
_DEICTIC_PATTERNS = tuple(_compile_phrase(p) for p in DEICTIC_PHRASES)
_IT_RE = re.compile(r"(?<!\w)it(?!\w)", flags=re.IGNORECASE)


def has_spec_intent(question: str) -> bool:
    text = question or ""
    return any(pattern.search(text) for _, pattern in _SPEC_PATTERNS)


def is_deictic(question: str) -> bool:
    text = question or ""
    if any(pattern.search(text) for pattern in _DEICTIC_PATTERNS):
        return True
    return bool(_IT_RE.search(text))


def derive_family(product_name: str) -> Optional[str]:
    """Brand/line name before finish or product descriptors."""
    raw = _strip_marks(product_name)
    if not raw:
        return None
    trimmed = _TRAILING_SKU_RE.sub("", raw).strip()
    lowered = trimmed.lower()
    for family in KNOWN_FAMILIES:
        key = family.lower()
        if family.lower() == "ben":
            if lowered == "ben" or lowered.startswith("ben "):
                return "ben"
            continue
        if lowered == key or lowered.startswith(key + " ") or lowered.startswith(key + "-"):
            if family in {"Ultra Spec EXT", "Ultra Spec Ext"}:
                return "Ultra Spec EXT"
            if family in {"INSL-X", "INSLX"}:
                return "INSL-X"
            if family in {"GarageGuard", "Garageguard"}:
                return "GarageGuard"
            if family == "Aura Bath & Spa":
                return "Aura"
            if family == "Eco Spec WB":
                return "Eco Spec"
            return family
    if lowered.startswith("hp ") or lowered.startswith("chp "):
        return "High Performance"
    tokens = lowered.split()
    kept: list[str] = []
    for token in tokens:
        bare = token.strip("-,/")
        if bare in _DESCRIPTOR_TOKENS:
            break
        kept.append(token)
    if kept:
        # Restore original capitalization from trimmed.
        original_tokens = trimmed.split()
        return " ".join(original_tokens[: len(kept)])
    return None


@dataclass
class Product:
    sku: str
    product_name: str
    discontinued: bool
    family: Optional[str]


@dataclass
class ProductRef:
    kind: str
    skus: list[str] = field(default_factory=list)
    sku: Optional[str] = None
    product_name: Optional[str] = None
    family: Optional[str] = None
    deictic: bool = False


@dataclass
class QueryPlan:
    action: str  # "filtered", "clarify", "unfiltered"
    ref: ProductRef
    spec_intent: bool
    deictic: bool
    clarify_text: str = ""


class ProductIndex:
    def __init__(self, products: list[Product]):
        self.products = products
        self.by_sku: dict[str, Product] = {p.sku: p for p in products}
        self.by_sku_upper: dict[str, Product] = {p.sku.upper(): p for p in products}
        self.family_to_skus: dict[str, list[str]] = {}
        for product in products:
            if not product.family:
                continue
            self.family_to_skus.setdefault(product.family, []).append(product.sku)
        self._numeric_sku_map = self._build_numeric_sku_map()
        self._name_entries = self._build_name_entries()
        self._family_entries = sorted(
            self.family_to_skus.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    def _build_numeric_sku_map(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for sku in self.by_sku:
            if sku.isdigit():
                mapping.setdefault(sku.lstrip("0") or "0", []).append(sku)
        return mapping

    def _build_name_entries(self) -> list[tuple[str, str, str]]:
        """(normalized_name, sku, original_name) longest first."""
        entries: list[tuple[str, str, str]] = []
        for product in self.products:
            stripped = _TRAILING_SKU_RE.sub("", _strip_marks(product.product_name)).strip()
            norm = normalize_name(stripped)
            if len(norm) < 8:
                continue
            entries.append((norm, product.sku, product.product_name))
        entries.sort(key=lambda item: len(item[0]), reverse=True)
        return entries

    def lookup_sku_token(self, token: str) -> Optional[Product]:
        raw = (token or "").strip()
        if not raw:
            return None
        exact = self.by_sku_upper.get(raw.upper())
        if exact:
            return exact
        if raw.isdigit():
            collapsed = raw.lstrip("0") or "0"
            hits = self._numeric_sku_map.get(collapsed, [])
            if len(hits) == 1:
                return self.by_sku[hits[0]]
        return None

    def resolve_product(self, question: str) -> ProductRef:
        deictic = is_deictic(question)
        sku_hits: list[Product] = []
        seen: set[str] = set()
        for token in _SKU_TOKEN_RE.findall(question or ""):
            product = self.lookup_sku_token(token)
            if product and product.sku not in seen:
                sku_hits.append(product)
                seen.add(product.sku)
        if len(sku_hits) == 1:
            product = sku_hits[0]
            return ProductRef(
                kind=KIND_SINGLE,
                skus=[product.sku],
                sku=product.sku,
                product_name=product.product_name,
                family=product.family,
                deictic=deictic,
            )
        if len(sku_hits) > 1:
            return ProductRef(
                kind=KIND_FAMILY,
                skus=[p.sku for p in sku_hits],
                family="matched SKUs",
                deictic=deictic,
            )

        qnorm = normalize_name(question or "")
        name_hits: list[tuple[str, str, str]] = []
        best_len = 0
        for norm, sku, original in self._name_entries:
            if norm in qnorm:
                if len(norm) > best_len:
                    name_hits = [(norm, sku, original)]
                    best_len = len(norm)
                elif len(norm) == best_len:
                    name_hits.append((norm, sku, original))
        unique_skus = list(dict.fromkeys(sku for _, sku, _ in name_hits))
        if len(unique_skus) == 1:
            product = self.by_sku[unique_skus[0]]
            return ProductRef(
                kind=KIND_SINGLE,
                skus=[product.sku],
                sku=product.sku,
                product_name=product.product_name,
                family=product.family,
                deictic=deictic,
            )

        text = question or ""
        for family, skus in self._family_entries:
            if family.lower() == "ben":
                pattern = _compile_phrase("ben")
            else:
                pattern = _compile_phrase(family)
            if pattern.search(text) and len(skus) >= 1:
                if len(skus) == 1:
                    product = self.by_sku[skus[0]]
                    return ProductRef(
                        kind=KIND_SINGLE,
                        skus=skus,
                        sku=product.sku,
                        product_name=product.product_name,
                        family=family,
                        deictic=deictic,
                    )
                return ProductRef(
                    kind=KIND_FAMILY,
                    skus=list(skus),
                    family=family,
                    deictic=deictic,
                )

        return ProductRef(kind=KIND_NONE, deictic=deictic)

    def clarification_text(self, ref: ProductRef) -> str:
        if ref.kind == KIND_FAMILY:
            lines = [
                "Which product do you mean? That name matches more than one product:"
            ]
            for sku in ref.skus[:10]:
                product = self.by_sku.get(sku)
                name = product.product_name if product else sku
                lines.append(f"- {name} ({sku})")
            if len(ref.skus) > 10:
                lines.append(f"- … and {len(ref.skus) - 10} more")
            return "\n".join(lines)
        return (
            "Which product do you mean? Please name the product or SKU."
        )

    def plan(self, question: str) -> QueryPlan:
        ref = self.resolve_product(question)
        spec = has_spec_intent(question)
        deictic = ref.deictic
        if ref.kind == KIND_SINGLE:
            return QueryPlan(action="filtered", ref=ref, spec_intent=spec, deictic=deictic)
        if ref.kind == KIND_FAMILY and spec:
            return QueryPlan(
                action="clarify",
                ref=ref,
                spec_intent=spec,
                deictic=deictic,
                clarify_text=self.clarification_text(ref),
            )
        if ref.kind == KIND_NONE and (spec or deictic):
            return QueryPlan(
                action="clarify",
                ref=ref,
                spec_intent=spec,
                deictic=deictic,
                clarify_text=self.clarification_text(ref),
            )
        return QueryPlan(action="unfiltered", ref=ref, spec_intent=spec, deictic=deictic)


def load_manifest_products(path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("classification") or "").strip() != "TDS":
                continue
            sku = (row.get("sku") or "").strip()
            if not sku:
                continue
            rows.append(row)
    return rows


def build_products(
    manifest_rows: Iterable[dict[str, str]],
) -> tuple[list[Product], list[dict[str, str]]]:
    products: list[Product] = []
    unassigned: list[dict[str, str]] = []
    for row in manifest_rows:
        sku = (row.get("sku") or "").strip()
        name = (row.get("product_name") or "").strip()
        discontinued = (row.get("discontinued") or "false").strip().lower() == "true"
        family = derive_family(name)
        product = Product(
            sku=sku,
            product_name=name,
            discontinued=discontinued,
            family=family,
        )
        products.append(product)
        if not family:
            unassigned.append({"sku": sku, "product_name": name})
    return products, unassigned


def write_families_csv(
    products: list[Product],
    path: Path = FAMILIES_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family", "sku", "product_name", "discontinued"],
        )
        writer.writeheader()
        for product in sorted(
            products,
            key=lambda p: ((p.family or "").lower(), p.sku.upper()),
        ):
            writer.writerow(
                {
                    "family": product.family or "",
                    "sku": product.sku,
                    "product_name": product.product_name,
                    "discontinued": "true" if product.discontinued else "false",
                }
            )


def load_index(
    manifest_path: Path = MANIFEST_PATH,
    families_path: Path = FAMILIES_PATH,
) -> ProductIndex:
    rows = load_manifest_products(manifest_path)
    products, _unassigned = build_products(rows)
    if not families_path.exists():
        write_families_csv(products, families_path)
    return ProductIndex(products)


_INDEX: Optional[ProductIndex] = None


def get_index() -> ProductIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = load_index()
    return _INDEX


def resolve_product(question: str) -> ProductRef:
    return get_index().resolve_product(question)


def plan_query(question: str) -> QueryPlan:
    return get_index().plan(question)


if __name__ == "__main__":
    rows = load_manifest_products()
    products, unassigned = build_products(rows)
    write_families_csv(products)
    print(f"wrote {FAMILIES_PATH} ({len(products)} products)")
    print(f"unassigned: {len(unassigned)}")
    for row in unassigned:
        print(f"  {row['sku']}\t{row['product_name']}")
    families = {p.family for p in products if p.family}
    print(f"families: {len(families)}")
