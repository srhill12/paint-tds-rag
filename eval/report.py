"""Print markdown eval tables from committed summary.json files.

README Evaluation numbers come from this output. Do not retype them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RESULTS_DIR  # noqa: E402

V1_RUNS = (
    ("Baseline", "20261007T204311Z_67c76a9"),
    ("Run 1 product-aware", "20261007T204608Z_67c76a9"),
    ("Run 2 router v2", "20261007T230030Z_5b07db3"),
    ("Run 3 full-sheet", "20261007T230816Z_20e3fd4"),
)

HOLDOUT_RUNS = (
    ("Router v1", "20261007T225628Z_5b07db3"),
    ("Router v2", "20261007T230010Z_5b07db3"),
)


def load_summary(run_id: str) -> dict:
    path = ROOT / RESULTS_DIR / run_id / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(item: dict | None) -> str:
    if not item:
        return "0/0"
    return f"{int(item.get('passed', 0))}/{int(item.get('total', 0))}"


def fmt_para(item: dict | None) -> str:
    if not item:
        return "0/0"
    return f"{int(item.get('routed', 0))}/{int(item.get('total', 0))}"


def keyword_safety(summary: dict) -> dict:
    """Non-paraphrased safety rows: overall routed minus paraphrased routed."""
    safety = summary["metrics"]["per_category"]["safety"]["safety_routed"]
    para = summary["metrics"]["paraphrased_safety"]
    passed = int(safety["passed"]) - int(para["routed"])
    total = int(safety["total"]) - int(para["total"])
    if passed < 0 or total < 0:
        raise SystemExit("keyword safety split produced a negative count")
    return {"passed": passed, "total": total}


def family_underspec(summary: dict) -> str:
    cat = summary["metrics"]["per_category"]
    family = fmt(cat["family"]["asks_which_product"])
    underspec = fmt(cat["underspecified"]["asks_which_product"])
    return f"{family}; {underspec}"


def v1_rows(summaries: list[dict]) -> list[tuple[str, list[str]]]:
    return [
        (
            "answerable passed",
            [
                fmt(s["metrics"]["per_category"]["answerable"]["passed"])
                for s in summaries
            ],
        ),
        (
            "answerable source-correct",
            [
                fmt(s["metrics"]["per_category"]["answerable"]["source_correct"])
                for s in summaries
            ],
        ),
        (
            "confusion wrong-product rate",
            [
                fmt(s["metrics"]["per_category"]["confusion"]["wrong_product_rate"])
                for s in summaries
            ],
        ),
        (
            "family and underspecified asked which product",
            [family_underspec(s) for s in summaries],
        ),
        (
            "safety routed (keyword)",
            [fmt(keyword_safety(s)) for s in summaries],
        ),
        (
            "safety routed (paraphrased)",
            [fmt_para(s["metrics"]["paraphrased_safety"]) for s in summaries],
        ),
        (
            "safety_negative not routed",
            [
                fmt(s["metrics"]["per_category"]["safety_negative"]["not_routed"])
                for s in summaries
            ],
        ),
        (
            "unanswerable passed",
            [
                fmt(s["metrics"]["per_category"]["unanswerable"]["passed"])
                for s in summaries
            ],
        ),
        (
            "hedged count",
            [fmt(s["metrics"]["hedged_count"]) for s in summaries],
        ),
        (
            "unsupported value rate",
            [fmt(s["metrics"]["unsupported_value_rate"]) for s in summaries],
        ),
        (
            "attribution rate",
            [fmt(s["metrics"]["attribution_rate"]) for s in summaries],
        ),
    ]


def holdout_rows(summaries: list[dict]) -> list[tuple[str, list[str]]]:
    return [
        (
            "safety routed",
            [
                fmt(s["metrics"]["per_category"]["safety"]["safety_routed"])
                for s in summaries
            ],
        ),
        (
            "safety_negative not routed",
            [
                fmt(s["metrics"]["per_category"]["safety_negative"]["not_routed"])
                for s in summaries
            ],
        ),
    ]


def markdown_table(headers: list[str], rows: list[tuple[str, list[str]]]) -> str:
    widths = [len(h) for h in headers]
    body = []
    for label, values in rows:
        cells = [label, *values]
        if len(cells) != len(headers):
            raise SystemExit("row width does not match headers")
        body.append(cells)
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    out = [line(headers), sep]
    out.extend(line(cells) for cells in body)
    return "\n".join(out)


def render() -> str:
    v1_summaries = [load_summary(run_id) for _, run_id in V1_RUNS]
    hold_summaries = [load_summary(run_id) for _, run_id in HOLDOUT_RUNS]
    n_v1 = v1_summaries[0]["n_questions"]
    n_hold = hold_summaries[0]["n_questions"]
    v1_ids = ", ".join(run_id for _, run_id in V1_RUNS)
    hold_ids = ", ".join(run_id for _, run_id in HOLDOUT_RUNS)

    v1_headers = ["Metric", *[label for label, _ in V1_RUNS]]
    hold_headers = ["Metric", *[label for label, _ in HOLDOUT_RUNS]]

    parts = [
        f"Test set (`n={n_v1}`). Source runs: {v1_ids}.",
        "",
        markdown_table(v1_headers, v1_rows(v1_summaries)),
        "",
        f"Router holdout (`n={n_hold}`). Source runs: {hold_ids}.",
        "",
        markdown_table(hold_headers, holdout_rows(hold_summaries)),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print markdown eval tables from summary.json files"
    )
    parser.parse_args()
    print(render(), end="")


if __name__ == "__main__":
    main()
