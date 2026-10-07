"""Re-score an existing eval run with the current scorer. Does not call the model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RESULTS_DIR, TESTSET_PATH  # noqa: E402
from eval.records import write_question_logs  # noqa: E402
from eval.scoring import SCORER_VERSION, score_row, summarize  # noqa: E402

COMMITTED_BASELINE = "20261007T201943Z_c01b8dc"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_testset(path: Path) -> dict[str, dict]:
    by_id = {}
    for row in load_jsonl(path):
        by_id[row["id"]] = row
    return by_id


def apply_score(item: dict, scored: dict) -> dict:
    updated = dict(item)
    updated["failure_reason"] = scored["failure_reason"]
    updated["passed"] = scored["passed"]
    updated["hedged"] = scored["hedged"]
    updated["numeric_match"] = scored["numeric_match"]
    updated["extracted_values"] = scored["quantities"]
    updated["value_support"] = scored.get("value_support") or []
    updated["unsupported_values"] = scored["unsupported_values"]
    updated["source_correct_values"] = scored.get("source_correct_values") or []
    updated["misattributed_values"] = scored.get("misattributed_values") or []
    updated["attribution"] = scored["attribution"]
    return updated


def rescore_run(run_dir: Path, testset_by_id: dict[str, dict]) -> dict:
    run_dir = run_dir.expanduser().resolve()
    committed = (ROOT / RESULTS_DIR / COMMITTED_BASELINE).resolve()
    if run_dir == committed:
        raise SystemExit(f"Refusing to modify committed baseline {run_dir}")

    pq_path = run_dir / "per_question.jsonl"
    summary_path = run_dir / "summary.json"
    if not pq_path.exists():
        raise SystemExit(f"missing {pq_path}")
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")

    items = load_jsonl(pq_path)
    scored_rows = []
    updated_items = []
    for item in items:
        qid = item.get("id")
        row = testset_by_id.get(qid)
        if row is None:
            raise SystemExit(f"{pq_path}: unknown id {qid}")
        scored = score_row(
            row,
            item.get("answer") or "",
            item.get("retrieved_skus") or [],
            item.get("retrieved_chunk_text") or [],
            bool(item.get("safety_routed")),
        )
        scored_rows.append(scored)
        updated_items.append(apply_score(item, scored))

    rows = [testset_by_id[item["id"]] for item in items]
    metrics = summarize(rows, scored_rows)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"] = metrics
    summary["scorer_version"] = SCORER_VERSION
    summary["rescored_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_question_logs(run_dir, updated_items)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score an eval run directory without calling the model"
    )
    parser.add_argument(
        "run_dir",
        nargs="+",
        type=Path,
        help="eval/results/<run_id> directory containing per_question.jsonl",
    )
    parser.add_argument(
        "--testset",
        type=Path,
        default=TESTSET_PATH,
        help="JSONL testset path (default: eval/testset_v1.jsonl)",
    )
    args = parser.parse_args()
    testset_path = args.testset
    if not testset_path.is_absolute():
        testset_path = ROOT / testset_path
    testset_by_id = load_testset(testset_path)

    for run_dir in args.run_dir:
        path = run_dir if run_dir.is_absolute() else ROOT / run_dir
        summary = rescore_run(path, testset_by_id)
        print(
            f"rescored {path} scorer_version={summary.get('scorer_version')} "
            f"rescored_at={summary.get('rescored_at')}"
        )


if __name__ == "__main__":
    main()
