import numpy as np

from vlmrl.envs import protocol

SCHEMA = {"left_arm": ["x", "y", "z"], "gripper": ["width"]}


def test_system_prompt_sections_and_substitution():
    sp = protocol.system_prompt("ENV TEXT", "do the thing", length_unit="centimeters")
    assert sp.index("## ENVIRONMENT") < sp.index("ENV TEXT") < sp.index("## TASK") < sp.index("do the thing")
    assert '{"<part>": {"<quantity>": <value>, ...}, ...}' in sp   # literal braces survive formatting
    assert "Lengths are in centimeters" in sp


def test_state_and_action_roundtrip():
    vals = [1.234, -5, 0.0, 8]
    assert protocol.format_state(SCHEMA, vals) == '{"left_arm": {"x": 1.23, "y": -5.0, "z": 0.0}, "gripper": {"width": 8.0}}'
    a = protocol.parse_action("reasoning...\n" + protocol.example_action(SCHEMA, vals), SCHEMA)
    assert np.allclose(a, [1.23, -5, 0, 8])


def test_parse_requires_every_quantity():
    assert protocol.parse_action('```json\n{"left_arm": {"x": 1, "y": 2, "z": 3}}\n```', SCHEMA) is None
    ok = protocol.parse_action('```json\n{"left_arm": {"x": 1, "y": 2, "z": 3}, "gripper": {"width": -1}}\n```', SCHEMA)
    assert np.allclose(ok, [1, 2, 3, -1])


def test_extract_prefers_last_fenced_block_and_handles_garbage():
    assert protocol.extract_action_json("```json\n{\"a\": 1}\n```\n```json\n{\"a\": 2}\n```") == {"a": 2}
    assert protocol.extract_action_json("```\n{\"a\": 3}\n```") == {"a": 3}
    assert protocol.extract_action_json("```json\n{not json\n```") is None
    assert protocol.extract_action_json("nothing") is None
