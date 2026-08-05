from scripts.myriad_eval.diagnose_physics_module import diagnose


def test_physics_module_is_connected_and_micro_overfits():
    result = diagnose(steps=20, learning_rate=3e-2, checkpoint=None)
    assert result["wiring"]["connected"]
    assert result["wiring"]["zero_bias_vs_disabled_max_abs"] == 0.0
    assert result["wiring"]["forced_bias_output_mean_abs_delta"] > 1e-5
    assert result["wiring"]["first_step_generator_grad_norm"] > 0.0
    assert result["micro_overfit"]["optimizable"]
    assert result["micro_overfit"]["loss_reduction_fraction"] >= 0.90
