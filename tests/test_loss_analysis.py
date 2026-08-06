import numpy as np

from scripts.myriad_eval.analyze_training_loss import series_summary


def test_series_summary_reports_window_change_and_slope():
    steps = np.arange(1, 11)
    values = np.arange(10.0, 0.0, -1.0)
    summary = series_summary(steps, values, window=2)
    assert summary["count"] == 10
    assert summary["first_window_mean"] == 9.5
    assert summary["last_window_mean"] == 1.5
    assert summary["linear_slope_per_1000_steps"] < 0
    assert summary["moving_average_min_step"] == 10
