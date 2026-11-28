"""scripts/probe_zero_shot.py: the zero-shot probe of a base model served by vLLM."""
import importlib.util
import os

from conftest import ROOT, make_traj


def _probe():
    spec = importlib.util.spec_from_file_location("probe_zero_shot", os.path.join(ROOT, "scripts", "probe_zero_shot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_sets():
    assert _probe().parse_sets("train_tasks=train:50, heldout_tasks=test:20") == [
        ("train_tasks", "train", 50), ("heldout_tasks", "test", 20)]


def test_summarize_reports_success_per_task_and_reply_length(tmp_path):
    probe = _probe()
    a, b, c = make_traj(success=True), make_traj(success=False, n_turns=2), make_traj(success=True)
    a.start_state, b.start_state, c.start_state = {"env_name": "reach-v3"}, {"env_name": "push-v3"}, {"env_name": "push-v3"}
    m = probe.summarize([a, b, c], "eval/train_tasks", object())
    assert m["eval/train_tasks/success"] == 2 / 3 and m["eval/train_tasks/episodes"] == 3
    assert m["eval/train_tasks/task/push-v3/success"] == 0.5
    assert m["eval/train_tasks/reply_chars"] == 5.0                        # "turn0", "turn1"
    probe.write_transcripts([a, b, c], str(tmp_path / "t.txt"), per_task=1)
    text = (tmp_path / "t.txt").read_text()
    assert text.count("===== push-v3") == 1 and "===== reach-v3" in text
