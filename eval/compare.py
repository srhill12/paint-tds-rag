"""Print a before/after table from two eval result directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scoring import CATEGORIES  # noqa: E402

GLOBAL_KEYS = (
    "hedged_count",
    "unsupported_value_rate",
    "attribution_rate",
)
VALUE_KEYS = (
    "source_correct",
    "misattributed",
    "unsupported",
    "attribution",
    "hedged",
)


def load_summary(directory: Path) -> dict:
    path = directory / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def frac(item: dict | None) -> tuple[int, int]:
    if not item:
        return 0, 0
    return int(item.get("passed", 0)), int(item.get("total", 0))


def fmt(item: dict | None) -> str:
    passed, total = frac(item)
    return f"{passed}/{total}"


def delta_str(before: dict | None, after: dict | None) -> str:
    b_p, b_t = frac(before)
    a_p, a_t = frac(after)
    if b_t == 0 and a_t == 0:
        return "0"
    if b_t == 0:
        return f"+{a_p}/{a_t}"
    change = a_p - b_p
    sign = "+" if change > 0 else ""
    return f"{sign}{change}"


def collect_rows(before: dict, after: dict) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    b_metrics = before.get("metrics") or {}
    a_metrics = after.get("metrics") or {}
    b_cat = b_metrics.get("per_category") or {}
    a_cat = a_metrics.get("per_category") or {}
    for category in CATEGORIES:
        b = b_cat.get(category) or {}
        a = a_cat.get(category) or {}
        keys = []
        for key, value in a.items():
            if key == "n":
                continue
            if isinstance(value, dict) and "passed" in value:
                keys.append(key)
        for key in keys:
            label = f"{category}.{key}"
            rows.append((label, fmt(b.get(key)), fmt(a.get(key)), delta_str(b.get(key), a.get(key))))
        for key in VALUE_KEYS:
            if key in keys:
                continue
            if key in b or key in a:
                label = f"{category}.{key}"
                rows.append(
                    (label, fmt(b.get(key)), fmt(a.get(key)), delta_str(b.get(key), a.get(key)))
                )
    for key in GLOBAL_KEYS:
        rows.append(
            (
                key,
                fmt(b_metrics.get(key)),
                fmt(a_metrics.get(key)),
                delta_str(b_metrics.get(key), a_metrics.get(key)),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval result directories")
    parser.add_argument("before", type=Path, help="Baseline result directory")
    parser.add_argument("after", type=Path, help="New result directory")
    args = parser.parse_args()
    before = load_summary(args.before)
    after = load_summary(args.after)
    print(
        f"before: {before.get('retrieval_mode')} {args.before}  "
        f"after: {after.get('retrieval_mode')} {args.after}"
    )
    print()
    print(f"{'metric':<42} {'before':>10} {'after':>10} {'delta':>8}")
    print("-" * 74)
    for label, b, a, delta in collect_rows(before, after):
        print(f"{label:<42} {b:>10} {a:>10} {delta:>8}")


if __name__ == "__main__":
    main()
