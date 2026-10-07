# Eval

Full per-question logs, including retrieved chunk text, are regenerated locally by running the eval after fetching the corpus. Public logs (`per_question_public.jsonl`) omit chunk text so the source documents are not republished. `summary.json` and `per_question_public.jsonl` are committable; `per_question.jsonl` is local-only.
