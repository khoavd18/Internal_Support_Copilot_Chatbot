# Retrieval And Answer Benchmark

The evaluation setup now has two layers:

- Retrieval benchmark
- Answer-quality benchmark built on the same case file

Both support two input formats:

- Preferred: `eval/queries/retrieval_benchmark.jsonl`
- Legacy fallback: `eval/queries/evaluation_queries.txt`

## JSONL Schema

Each line is one benchmark case:

```json
{
  "id": "passkey-sign-in",
  "query": "How to sign in with a passkey?",
  "expected_sources": ["github_docs"],
  "expected_documents": ["signing-in-with-a-passkey"],
  "expected_categories": ["authentication"],
  "expected_answer_intent": "github_authentication",
  "notes": "Should rank GitHub Docs passkey guidance near the top.",
  "answer_quality": {
    "reference_answer": "Explain how to sign in with a passkey and cite GitHub Docs guidance.",
    "must_include": ["passkey", "sign in"],
    "must_not_include": ["ssh key"],
    "completeness_points": ["passkey", "sign in", "github"],
    "minimum_source_count": 1
  }
}
```

Field notes:

- `query`: required
- `expected_sources`: optional list of expected `source` values such as `github_docs`
- `expected_documents`: optional list of case-insensitive fragments matched against `doc_id`, `origin_doc_id`, `title`, `path`, or `url`
- `expected_categories`: optional list of case-insensitive fragments matched against `category`, `source_type`, `title`, or `path`
- `expected_answer_intent`: optional string or list of strings matched against the query analyzer intent labels
- `notes`: optional free-form comment for maintainers
- `answer_quality.reference_answer`: optional short reference answer used for lightweight keyword-recall checking
- `answer_quality.must_include`: optional phrases that should appear in a good answer
- `answer_quality.must_not_include`: optional phrases that should not appear in a good answer
- `answer_quality.completeness_points`: optional checklist of important answer points
- `answer_quality.minimum_source_count`: optional minimum number of returned sources expected for a grounded answer

Top-level aliases such as `reference_answer` and `answer_must_include` are also accepted, but the nested `answer_quality` object is the preferred format.

## Retrieval Outputs

Running the retrieval benchmark writes:

- `eval/runs/<timestamp>_retrieval_benchmark.jsonl`: per-query results with retrieved rows and matches
- `eval/runs/<timestamp>_retrieval_summary.json`: machine-readable aggregate summary
- `eval/reports/latest_retrieval_summary.txt`: latest human-readable report

## Answer-Quality Outputs

Running the answer-quality benchmark writes:

- `eval/runs/<timestamp>_answer_quality_benchmark.jsonl`: per-case retrieval context, generated answer, and dimension-level scores
- `eval/runs/<timestamp>_answer_quality_summary.json`: machine-readable answer-quality summary
- `eval/reports/<timestamp>_answer_quality_summary.txt`: timestamped human-readable answer-quality report
- `eval/reports/latest_answer_quality_summary.txt`: latest human-readable answer-quality report

## Commands

```bash
python scripts/dev.py benchmark-retrieval
```

```bash
python scripts/dev.py benchmark-answers
```

Optional flags are passed through to the benchmark script:

```bash
python scripts/dev.py benchmark-retrieval -- --top-k 8 --queries-path eval/queries/retrieval_benchmark.jsonl
```

```bash
python scripts/dev.py benchmark-answers -- --top-k 5 --queries-path eval/queries/retrieval_benchmark.jsonl
```

## Adding New Cases

1. Add a new JSON line to `eval/queries/retrieval_benchmark.jsonl`.
2. Fill in the retrieval labels you care about: source, document, category, or intent.
3. Add an `answer_quality` object only for the dimensions you want to evaluate.
4. Keep rubric phrases short, explicit, and easy to match literally.
5. Prefer stable concepts over exact sentence wording so small model phrasing changes do not invalidate the case.
