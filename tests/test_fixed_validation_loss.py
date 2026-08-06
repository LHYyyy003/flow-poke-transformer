from scripts.myriad_eval.evaluate_fixed_validation_loss import summarize


def test_summarize_reports_mean_and_standard_error():
    result = summarize([1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 4
    assert result["mean"] == 2.5
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["standard_error"] > 0
