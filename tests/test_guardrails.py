from src.agent.guardrails import evaluate_answer_guardrails


def test_guardrail_clarify_when_no_docs():
    result = {
        "sources": [],
        "stats": {"retrieved_docs": 0},
    }
    decision = evaluate_answer_guardrails(result, requested_top_k=4)
    assert decision.action == "fallback_to_clarify"


def test_guardrail_allow_when_scores_good():
    result = {
        "sources": [
            {"rerank_score": 8.2},
            {"rerank_score": 6.9},
        ],
        "stats": {"retrieved_docs": 2},
    }
    decision = evaluate_answer_guardrails(result, requested_top_k=4)
    assert decision.action == "allow_answer"