"""
Fetch Benjamin Moore Technical Data Sheets listed in sources/manifest.csv
Downloads each tds_url into pdfs/, one request per second, and skips files
that already exist.

With --record-hashes, writes the MD5 of each local file into manifest.csv.
Without the flag, compares a newly downloaded file to the manifest MD5 and
warns if the manufacturer has revised the sheet since this corpus was evaluated.
"""

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote, unquote, urlparse, urlsplit, urlunsplit

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "sources" / "manifest.csv"
PDF_DIR       = REPO_ROOT / "pdfs"
USER_AGENT    = "Mozilla/5.0 (paint-tds-rag source fetcher)"
TIMEOUT_SECS  = 60
REQUEST_INTERVAL_SECS = 1

REVISED_WARNING = (
    "the manufacturer has revised the sheet since this corpus was evaluated."
)


def target_filename(url: str, sku: str) -> str:
    """Use the PDF name from the URL; fall back to the SKU when the URL has none."""
    name = Path(unquote(urlparse(url).path)).name
    if name.lower().endswith(".pdf"):
        return name
    return f"{sku}.pdf"


def encode_download_url(url: str) -> str:
    """URL-encode spaces and other unsafe characters in the path."""
    parts = urlsplit(url)
    path = quote(unquote(parts.path), safe="/")
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download to a temporary file, verify it is a PDF, then move into place."""
    request = urllib.request.Request(
        encode_download_url(url),
        headers={"User-Agent": USER_AGENT},
    )
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


def write_manifest(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download TDS PDFs listed in sources/manifest.csv"
    )
    parser.add_argument(
        "--record-hashes",
        action="store_true",
        help="Write the MD5 of each downloaded (or already local) file into manifest.csv",
    )
    args = parser.parse_args(argv)

    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return 1

    PDF_DIR.mkdir(exist_ok=True)

    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    downloaded, skipped, failed = 0, 0, 0
    failures: list[str] = []
    last_request_at: float | None = None

    for row in rows:
        url = (row.get("tds_url") or "").strip()
        sku = (row.get("sku") or "").strip()
        product = (row.get("product_name") or "").strip()
        filename = (row.get("filename") or "").strip()
        expected_md5 = (row.get("md5") or "").strip()

        if not url.startswith(("http://", "https://")):
            print(f"  No tds_url for {product or sku or filename or 'unnamed row'}, skipping.")
            skipped += 1
            continue

        dest = PDF_DIR / (filename or target_filename(url, sku))
        if dest.exists():
            print(f"  {dest.name} already exists, skipping.")
            skipped += 1
            if args.record_hashes:
                row["md5"] = md5_file(dest)
            continue

        if last_request_at is not None:
            wait = REQUEST_INTERVAL_SECS - (time.monotonic() - last_request_at)
            if wait > 0:
                time.sleep(wait)

        try:
            download(url, dest)
            last_request_at = time.monotonic()
            print(f"  Downloaded {dest.name}")
            downloaded += 1
            actual_md5 = md5_file(dest)
            if args.record_hashes:
                row["md5"] = actual_md5
            elif expected_md5 and actual_md5 != expected_md5:
                print(
                    f"  Warning: {dest.name} MD5 {actual_md5} does not "
                    f"match manifest {expected_md5}. {REVISED_WARNING}"
                )
        except Exception as e:
            last_request_at = time.monotonic()
            print(f"  Failed {dest.name}: {e}")
            failed += 1
            failures.append(f"{dest.name}: {e}")

    if args.record_hashes:
        if "md5" not in fieldnames:
            fieldnames.append("md5")
        write_manifest(rows, fieldnames)

    print()
    print(f"Downloaded: {downloaded}  Skipped: {skipped}  Failed: {failed}")
    if failures:
        print("Failures:")
        for item in failures:
            print(f"  {item}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
