"""
Fetch Benjamin Moore Technical Data Sheets listed in sources/manifest.csv
Downloads each tds_url into pdfs/ and skips files that already exist.
"""

import csv
import sys
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "sources" / "manifest.csv"
PDF_DIR       = REPO_ROOT / "pdfs"
USER_AGENT    = "Mozilla/5.0 (paint-tds-rag source fetcher)"
TIMEOUT_SECS  = 60

def target_filename(url: str, sku: str) -> str:
    """Use the PDF name from the URL; fall back to the SKU when the URL has none."""
    name = Path(unquote(urlparse(url).path)).name
    if name.lower().endswith(".pdf"):
        return name
    return f"{sku}.pdf"

def download(url: str, dest: Path) -> None:
    """Download to a temporary file, verify it is a PDF, then move into place."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp_path = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECS) as response:
            data = response.read()
        if not data.startswith(b"%PDF"):
            raise ValueError("response is not a PDF")
        tmp_path.write_bytes(data)
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)

def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return 1

    PDF_DIR.mkdir(exist_ok=True)

    downloaded, skipped, failed = 0, 0, 0
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("tds_url") or "").strip()
            sku = (row.get("sku") or "").strip()
            product = (row.get("product_name") or "").strip()

            if not url.startswith(("http://", "https://")):
                print(f"  No tds_url for {product or sku or 'unnamed row'}, skipping.")
                skipped += 1
                continue

            dest = PDF_DIR / target_filename(url, sku)
            if dest.exists():
                print(f"  {dest.name} already exists, skipping.")
                skipped += 1
                continue

            try:
                download(url, dest)
                print(f"  Downloaded {dest.name}")
                downloaded += 1
            except Exception as e:
                print(f"  Failed {url}: {e}")
                failed += 1

    print()
    print(f"Downloaded: {downloaded}  Skipped: {skipped}  Failed: {failed}")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
