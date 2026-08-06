from scripts.myriad_eval.evaluate_physics_multiscene import SCENE_SPECS, build_scene


def test_fixed_multiscene_protocol_has_expected_events():
    scenes = {spec.name: build_scene(spec) for spec in SCENE_SPECS}
    assert set(scenes) == {"head_on", "oblique", "grazing", "no_collision", "wall_bounce"}
    for name in ("head_on", "oblique", "grazing"):
        assert scenes[name].ball_collision_steps
    assert not scenes["no_collision"].any_collision_steps
    assert scenes["wall_bounce"].any_collision_steps
    assert not scenes["wall_bounce"].ball_collision_steps
