"""Run the Paint TDS eval harness.

Draft mode (--draft) prints metrics and does not write to eval/results/.
--draft-out DIR writes per_question.jsonl and summary.json outside eval/results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# FAISS + Anaconda OpenMP on macOS abort without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
    PRIOR_TEMPERATURE,
    RESULTS_DIR,
    RETRIEVAL_MODE,
    RETRIEVAL_MODES,
    SEED,
    TEMPERATURE,
    TESTSET_PATH,
    TESTSET_VERSION,
    TOP_K,
)
from eval.records import chunk_ids_for, write_question_logs  # noqa: E402
from eval.scoring import CATEGORIES, TESTSET_FIELDS, score_row, summarize  # noqa: E402
from rag import load_vector_store, run_query  # noqa: E402


def load_testset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [field for field in TESTSET_FIELDS if field not in row]
            if missing:
                raise SystemExit(f"{path}:{line_no} missing fields {missing}")
            rows.append(row)
    return rows


def unverified_ids(rows: list[dict]) -> list[str]:
    return [
        row.get("id", "")
        for row in rows
        if not str(row.get("verified_by") or "").strip()
    ]


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_ollama_list() -> dict[str, str]:
    """Map model name -> digest/id from `ollama list`."""
    try:
        output = subprocess.check_output(["ollama", "list"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return {}
    mapping: dict[str, str] = {}
    lines = output.strip().splitlines()
    if not lines:
        return mapping
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[0] not in mapping:
            mapping[parts[0]] = parts[1]
    return mapping


def ollama_digest(mapping: dict[str, str], name: str) -> str:
    if name in mapping:
        return mapping[name]
    latest = f"{name}:latest"
    if latest in mapping:
        return mapping[latest]
    bare = name.split(":")[0]
    for key, digest in mapping.items():
        if key.split(":")[0] == bare:
            return digest
    return ""


def corpus_retrieved_date(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        dates = {
            (row.get("retrieved_date") or "").strip()
            for row in reader
            if (row.get("retrieved_date") or "").strip()
        }
    if not dates:
        return ""
    if len(dates) == 1:
        return dates.pop()
    return ",".join(sorted(dates))


def source_skus(source_docs) -> list[str]:
    return [(doc.metadata.get("sku") or "").strip() for doc in source_docs]


def source_products(source_docs) -> list[str]:
    products = []
    for doc in source_docs:
        name = (
            doc.metadata.get("product_name")
            or doc.metadata.get("product")
            or ""
        ).strip()
        products.append(name)
    return products


def source_chunks(source_docs) -> list[str]:
    return [doc.page_content for doc in source_docs]


def source_scores(source_docs) -> list:
    scores = []
    for doc in source_docs:
        score = doc.metadata.get("score")
        scores.append(score)
    return scores


def format_frac(item: dict) -> str:
    total = item.get("total", 0)
    passed = item.get("passed", 0)
    return f"{passed}/{total}"


def print_report(summary: dict, prior_temperature: float, retrieval_mode: str = "") -> None:
    print(f"prior_temperature: {prior_temperature} -> temperature: {TEMPERATURE}")
    print(f"seed: {SEED}  top_k: {TOP_K}  model: {CHAT_MODEL}")
    if retrieval_mode:
        print(f"retrieval_mode: {retrieval_mode}")
    print()
    print("Per-category results (counts / denominators):")
    for category, metrics in summary["per_category"].items():
        parts = [f"{category}: passed {format_frac(metrics['passed'])}"]
        for key, value in metrics.items():
            if key in {"n", "passed"}:
                continue
            if isinstance(value, dict) and "passed" in value:
                parts.append(f"{key} {format_frac(value)}")
        print("  " + "; ".join(parts))
    print()
    print(f"hedged_count: {format_frac(summary['hedged_count'])}")
    unsupported = summary["unsupported_value_rate"]
    print(f"unsupported_value_rate: {format_frac(unsupported)}")
    for item in unsupported.get("items", []):
        print(f"  {item['id']}: {item['values']}")
    print(f"attribution_rate: {format_frac(summary['attribution_rate'])}")
    para = summary["paraphrased_safety"]
    print(f"paraphrased_safety_routed: {para['routed']}/{para['total']}")
    print()
    print("Misattributed values (id, value, attesting SKUs):")
    misattributed = summary.get("misattributed_values") or []
    if not misattributed:
        print("  (none)")
    for item in misattributed:
        print(f"  {item['id']}: {item['value']} <- {item['attesting_skus']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Paint TDS eval harness")
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Allow unverified rows; print results and do not write to eval/results/",
    )
    parser.add_argument(
        "--draft-out",
        type=Path,
        default=None,
        help="Write draft per_question.jsonl and summary.json to this directory "
        "(must be outside eval/results/)",
    )
    parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODES,
        default=None,
        help="Retrieval mode (default: config RETRIEVAL_MODE)",
    )
    parser.add_argument(
        "--testset",
        type=Path,
        default=TESTSET_PATH,
        help="JSONL testset path (default: eval/testset_v1.jsonl)",
    )
    args = parser.parse_args()
    draft_mode = bool(args.draft or args.draft_out)
    retrieval_mode = args.mode or RETRIEVAL_MODE
    testset_path = args.testset
    if not testset_path.is_absolute():
        testset_path = ROOT / testset_path

    rows = load_testset(testset_path)
    missing_verification = unverified_ids(rows)
    if missing_verification and not draft_mode:
        preview = ", ".join(missing_verification[:10])
        raise SystemExit(
            f"Refusing to run {testset_path}: "
            f"{len(missing_verification)} row(s) have empty verified_by "
            f"(e.g. {preview}). Pass --draft to print results without writing."
        )
    draft_out_dir = None
    if args.draft_out is not None:
        draft_out_dir = args.draft_out.expanduser().resolve()
        results_root = (ROOT / RESULTS_DIR).resolve()
        if draft_out_dir == results_root or results_root in draft_out_dir.parents:
            raise SystemExit("--draft-out must be a directory outside eval/results/")

    resources = load_vector_store()
    if resources is None:
        raise SystemExit("Vector store not found or failed to load. Is Ollama running?")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit_hash()
    run_id = f"{timestamp}_{commit[:7]}"
    ollama_digests = parse_ollama_list()

    scored_rows = []
    per_question = []
    for row in rows:
        result = run_query(
            row["question"], resources, retrieval_mode=retrieval_mode
        )
        docs = result.get("source_docs") or []
        skus = source_skus(docs)
        products = source_products(docs)
        chunks = source_chunks(docs)
        chunk_ids = chunk_ids_for(skus, chunks)
        answer = result.get("answer") or ""
        label = result.get("product_label") or ""
        if label:
            answer = f"{label}\n{answer}"
        scored = score_row(
            row,
            answer,
            skus,
            chunks,
            bool(result.get("safety_routed")),
            retrieved_products=products,
        )
        scored_rows.append(scored)
        per_question.append(
            {
                "id": row["id"],
                "question": row["question"],
                "category": row["category"],
                "answer": answer,
                "retrieved_skus": skus,
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_chunk_text": chunks,
                "safety_routed": bool(result.get("safety_routed")),
                "matched_terms": result.get("matched_terms") or [],
                "scores": source_scores(docs),
                "failure_reason": scored["failure_reason"],
                "passed": scored["passed"],
                "hedged": scored["hedged"],
                "numeric_match": scored["numeric_match"],
                "extracted_values": scored["quantities"],
                "value_support": scored.get("value_support") or [],
                "unsupported_values": scored["unsupported_values"],
                "source_correct_values": scored.get("source_correct_values") or [],
                "misattributed_values": scored.get("misattributed_values") or [],
                "attribution": scored["attribution"],
            }
        )
        status = "PASS" if scored["passed"] else "FAIL"
        reason = scored["failure_reason"] or "ok"
        print(f"[{status}] {row['id']} ({row['category']}) {reason}")

    metrics = summarize(rows, scored_rows)
    summary = {
        "run_id": run_id,
        "git_commit": commit,
        "model_name": CHAT_MODEL,
        "model_digest": ollama_digest(ollama_digests, CHAT_MODEL),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_digest": ollama_digest(ollama_digests, EMBEDDING_MODEL),
        "temperature": TEMPERATURE,
        "prior_temperature": PRIOR_TEMPERATURE,
        "seed": SEED,
        "top_k": TOP_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "testset_version": TESTSET_VERSION,
        "corpus_retrieved_date": corpus_retrieved_date(MANIFEST_PATH),
        "timestamp": timestamp,
        "retrieval_mode": retrieval_mode,
        "draft": draft_mode,
        "n_questions": len(rows),
        "metrics": metrics,
    }

    print()
    print_report(metrics, PRIOR_TEMPERATURE, retrieval_mode=retrieval_mode)

    if draft_mode and draft_out_dir is None:
        print("\nDraft run: nothing written to eval/results/.")
        return

    out_dir = draft_out_dir if draft_out_dir is not None else ROOT / RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_question_logs(out_dir, per_question)
    if draft_mode:
        print(f"\nDraft run: wrote {out_dir} (not eval/results/).")
    else:
        print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
