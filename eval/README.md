# Eval

Full per-question logs, including retrieved chunk text, are regenerated locally by running the eval after fetching the corpus. Public logs (`per_question_public.jsonl`) omit chunk text so the source documents are not republished. `summary.json` and `per_question_public.jsonl` are committable; `per_question.jsonl` is local-only.

The holdout questions in `testset_router_holdout.jsonl` were drafted in the same session as the v2 term list, so they are not fully independent of it. Results on the holdout are a check on false positives and obvious misses, not proof of real-world recall.
