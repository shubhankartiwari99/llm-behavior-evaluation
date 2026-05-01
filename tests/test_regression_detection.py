from scripts.regression_engine import (
    compare_models,
    decide_promotion,
    evaluate_model,
    load_dataset,
    load_model,
)


def test_evaluate_model_and_compare():
    dataset = load_dataset("data/eval_dataset.json")
    prod_model = load_model("models/model_v1.json")
    cand_model = load_model("models/model_v2.json")

    prod_metrics = evaluate_model(prod_model, dataset)
    cand_metrics = evaluate_model(cand_model, dataset)
    deltas = compare_models(prod_metrics, cand_metrics)

    assert prod_metrics.accuracy == 1.0
    assert prod_metrics.hallucination_rate == 0.0
    assert prod_metrics.format_error_rate == 0.0

    assert cand_metrics.accuracy < prod_metrics.accuracy
    assert cand_metrics.hallucination_rate > prod_metrics.hallucination_rate
    assert cand_metrics.format_error_rate > prod_metrics.format_error_rate

    assert deltas["delta_accuracy"] < 0
    assert deltas["delta_hallucination"] > 0
    assert deltas["delta_format_error"] > 0


def test_decision_policy_rejects_candidate_with_regressions():
    dataset = load_dataset("data/eval_dataset.json")
    prod_model = load_model("models/model_v1.json")
    cand_model = load_model("models/model_v2.json")

    prod_metrics = evaluate_model(prod_model, dataset)
    cand_metrics = evaluate_model(cand_model, dataset)
    decision = decide_promotion(prod_metrics, cand_metrics)

    assert decision.decision == "reject"
    assert decision.reason in {
        "hallucination_increased",
        "format_error_increased",
        "accuracy_decreased",
    }
