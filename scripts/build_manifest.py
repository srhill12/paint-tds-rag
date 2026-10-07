"""
Build sources/manifest.csv as a dated snapshot of the Benjamin Moore US TDS index.

Fetches the public listing once, keeps every unique .pdf link (deduped by URL,
.zip ignored), and writes provenance columns. Existing md5, retrieved_date,
and rows marked classification_source=manual are preserved on regenerate.
New rows have empty md5 until scripts/fetch_sources.py --record-hashes.
"""

import csv
import re
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

INDEX_URL = "https://www.benjaminmoore.com/en-us/data-sheets/technical-data-sheets"
USER_AGENT = "Mozilla/5.0 (paint-tds-rag source fetcher)"
TIMEOUT_SECS = 60

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "sources" / "manifest.csv"

COLUMNS = [
    "filename",
    "product_name",
    "sku",
    "discontinued",
    "classification",
    "classification_source",
    "tds_url",
    "md5",
    "retrieved_date",
]

# Trailing product codes must contain a digit or an XX placeholder so
# titles that end in "Semi-Gloss" are not treated as SKUs.
_SKU_CODE = (
    r"(?:"
    r"(?=[A-Z0-9./-]*\d)[A-Z0-9]+(?:[./-][A-Z0-9]+)*"  # AC-01, 015-00, 1PR.100
    r"|[A-Z]+-X+"  # EGF-XXX, HTF-XXX
    r")(?:\s+Line)?"
)
SKU_RE = re.compile(
    rf"(?:{_SKU_CODE})(?:\s*,\s*{_SKU_CODE})*\s*$"
)

NON_TDS_TERMS = ("guide", "faq", "test", "glossary", "brochure", "solver")


class AnchorParser(HTMLParser):
    """Collect (title, href) for every HTML anchor."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._href = href
                self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            title = " ".join("".join(self._parts).split())
            self.links.append((title, self._href))
            self._href = None
            self._parts = []


def fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=TIMEOUT_SECS) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_index_pdfs(html: str, base_url: str) -> dict[str, str]:
    """Return unique media.benjaminmoore.com PDF URLs mapped to link title."""
    parser = AnchorParser()
    parser.feed(html)
    seen: dict[str, str] = {}
    for title, href in parser.links:
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        host = parsed.netloc.lower()
        path = parsed.path.lower()
        if "media.benjaminmoore.com" not in host:
            continue
        if path.endswith(".zip"):
            continue
        if not path.endswith(".pdf"):
            continue
        if full not in seen:
            seen[full] = title
    return seen


def url_basename(url: str) -> str:
    return unquote(urlparse(url).path.rsplit("/", 1)[-1])


def parse_title(title: str) -> tuple[str, str, bool]:
    """Return (product_name, sku, discontinued) from an index link title."""
    discontinued = bool(re.search(r"\(discontinued\)", title, flags=re.I))
    product_name = re.sub(r"\s*\(discontinued\)\s*", " ", title, flags=re.I)
    product_name = " ".join(product_name.split())
    sku_source = re.sub(r"\s*\*.*$", "", product_name)
    match = SKU_RE.search(sku_source)
    sku = match.group(0).strip() if match else ""
    return product_name, sku, discontinued


def classify_from_title_or_filename(title: str, filename: str) -> str:
    """NON_TDS if the link title or filename contains a non-sheet keyword."""
    blob = f"{title} {filename}".lower()
    if any(term in blob for term in NON_TDS_TERMS):
        return "NON_TDS"
    return "TDS"


def load_existing_rows(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Index existing manifest rows by URL and filename so hashes can be kept."""
    by_url: dict[str, dict[str, str]] = {}
    by_name: dict[str, dict[str, str]] = {}
    if not path.exists():
        return by_url, by_name
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            url = (row.get("tds_url") or "").strip()
            name = (row.get("filename") or "").strip()
            if url:
                by_url[url] = row
            if name:
                by_name[name] = row
    return by_url, by_name


def main() -> int:
    print(f"Fetching {INDEX_URL}")
    html = fetch_html(INDEX_URL)
    index_pdfs = parse_index_pdfs(html, INDEX_URL)
    existing_by_url, existing_by_name = load_existing_rows(MANIFEST_PATH)

    today = date.today().isoformat()
    rows: list[dict[str, str]] = []
    for url, title in index_pdfs.items():
        filename = url_basename(url)
        product_name, sku, discontinued = parse_title(title)
        existing = existing_by_url.get(url) or existing_by_name.get(filename) or {}
        if (existing.get("classification_source") or "").strip().lower() == "manual":
            classification = (existing.get("classification") or "").strip() or "TDS"
            classification_source = "manual"
        else:
            classification = classify_from_title_or_filename(title, filename)
            classification_source = "auto"
        rows.append(
            {
                "filename": filename,
                "product_name": product_name,
                "sku": sku,
                "discontinued": "true" if discontinued else "false",
                "classification": classification,
                "classification_source": classification_source,
                "tds_url": url,
                "md5": (existing.get("md5") or "").strip(),
                "retrieved_date": (existing.get("retrieved_date") or "").strip() or today,
            }
        )

    rows.sort(key=lambda row: row["filename"].lower())

    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    tds = sum(1 for row in rows if row["classification"] == "TDS")
    non_tds = sum(1 for row in rows if row["classification"] == "NON_TDS")
    discontinued = sum(1 for row in rows if row["discontinued"] == "true")

    print(f"Wrote {MANIFEST_PATH} ({len(rows)} rows)")
    print(f"TDS: {tds}")
    print(f"NON_TDS: {non_tds}")
    print(f"Discontinued: {discontinued}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
