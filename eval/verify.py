"""Walk unverified testset rows and confirm them against the source PDF."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TESTSET_PATH  # noqa: E402
from eval.scoring import TESTSET_FIELDS  # noqa: E402

VERIFIER_NAME = "Steven Hill"
EDITABLE_FIELDS = (
    "expected_value",
    "expected_unit",
    "expected_sku",
    "acceptable_skus",
    "source_file",
    "source_page",
    "evidence_quote",
)
DISPLAY_FIELDS = (
    "id",
    "category",
    "question",
    "expected_value",
    "expected_unit",
    "expected_sku",
    "source_file",
    "source_page",
    "evidence_quote",
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pdf_path_for(source_file: str) -> Path | None:
    if not source_file:
        return None
    name = Path(source_file).with_suffix(".pdf").name
    candidates = [
        ROOT / "pdfs" / name,
        ROOT / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def open_pdf(source_file: str) -> None:
    path = pdf_path_for(source_file)
    if path is None:
        return
    if not path.exists():
        print(f"  PDF not found: {path}")
        return
    try:
        subprocess.run(["open", str(path)], check=False)
        print(f"  opened {path}")
    except OSError as exc:
        print(f"  could not open PDF: {exc}")


def prompt_choice() -> str:
    while True:
        raw = input("[y] verify as-is, [e] edit expected fields then verify, [s] skip: ").strip().lower()
        if raw in {"y", "e", "s"}:
            return raw
        print("  enter y, e, or s")


def prompt_edit(row: dict) -> dict:
    print("  Enter a new value, or press return to keep the current one.")
    originals = {field: row.get(field) for field in EDITABLE_FIELDS}
    changed: list[str] = []
    for field in EDITABLE_FIELDS:
        current = row.get(field)
        shown = current if not isinstance(current, list) else ", ".join(str(x) for x in current)
        typed = input(f"  {field} [{shown}]: ")
        if typed == "":
            continue
        if field == "acceptable_skus":
            row[field] = [part.strip() for part in typed.split(",") if part.strip()]
        else:
            row[field] = typed
        changed.append(field)
    if changed:
        originals_shown = {field: originals[field] for field in changed}
        note = (
            "corrected during verification from "
            f"{json.dumps(originals_shown, ensure_ascii=False)}; basis: PDF"
        )
        existing = (row.get("notes") or "").strip()
        row["notes"] = f"{existing} {note}".strip() if existing else note
    return row


def mark_verified(row: dict) -> dict:
    row["verified_by"] = f"{VERIFIER_NAME} {date.today().isoformat()}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Paint TDS eval testset rows against source PDFs"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=TESTSET_PATH,
        help="JSONL testset path (default: eval/testset_v1.jsonl)",
    )
    args = parser.parse_args()
    path = args.file
    if not path.is_absolute():
        path = ROOT / path
    rows = load_rows(path)
    pending = [i for i, row in enumerate(rows) if not str(row.get("verified_by") or "").strip()]
    print(f"{len(pending)} unverified row(s) in {path}")
    if not pending:
        return

    for index in pending:
        row = rows[index]
        print()
        print("=" * 72)
        for field in DISPLAY_FIELDS:
            print(f"{field}: {row.get(field)}")
        missing = [field for field in TESTSET_FIELDS if field not in row]
        if missing:
            print(f"  warning: missing fields {missing}")
        if row.get("source_file"):
            open_pdf(str(row["source_file"]))
        choice = prompt_choice()
        if choice == "s":
            print("  skipped")
            continue
        if choice == "e":
            row = prompt_edit(row)
        rows[index] = mark_verified(row)
        save_rows(path, rows)
        print(f"  saved {path} ({row['id']} verified_by={row['verified_by']})")


if __name__ == "__main__":
    main()
