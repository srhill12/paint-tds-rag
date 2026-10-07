"""Eval result records. Public logs omit retrieved chunk text."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def make_chunk_id(sku: str, text: str) -> str:
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
    prefix = (sku or "").strip() or "chunk"
    return f"{prefix}:{digest}"


def chunk_ids_for(skus: list[str], texts: list[str]) -> list[str]:
    n = max(len(skus), len(texts), 0)
    ids = []
    for i in range(n):
        sku = skus[i] if i < len(skus) else ""
        text = texts[i] if i < len(texts) else ""
        ids.append(make_chunk_id(sku, text))
    return ids


def _extracted_public(item: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    support = item.get("value_support") or []
    if support:
        for decision in support:
            out.append(
                {
                    "value": decision.get("value"),
                    "supported": decision.get("supported"),
                    "source": decision.get("source"),
                    "attesting_skus": list(decision.get("attesting_skus") or []),
                }
            )
        return out
    for raw in item.get("extracted_values") or []:
        out.append(
            {
                "value": raw,
                "supported": None,
                "source": "",
                "attesting_skus": [],
            }
        )
    return out


def to_public_record(item: dict) -> dict[str, Any]:
    """Drop retrieved chunk text; keep SKUs, chunk ids, and attestation labels."""
    skus = list(item.get("retrieved_skus") or [])
    texts = list(item.get("retrieved_chunk_text") or [])
    ids = item.get("retrieved_chunk_ids")
    if not ids:
        ids = chunk_ids_for(skus, texts)
    else:
        ids = list(ids)
    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "retrieved_skus": skus,
        "retrieved_chunk_ids": ids,
        "extracted_values": _extracted_public(item),
        "safety_routed": bool(item.get("safety_routed")),
        "matched_terms": list(item.get("matched_terms") or []),
        "scores": list(item.get("scores") or []),
        "failure_reason": item.get("failure_reason") or "",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_question_logs(out_dir: Path, items: list[dict]) -> None:
    """Write private per_question.jsonl and public per_question_public.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "per_question.jsonl", items)
    write_jsonl(
        out_dir / "per_question_public.jsonl",
        [to_public_record(item) for item in items],
    )


def write_public_log(out_dir: Path, items: list[dict]) -> None:
    """Write only the public log; leave per_question.jsonl unchanged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        out_dir / "per_question_public.jsonl",
        [to_public_record(item) for item in items],
    )
