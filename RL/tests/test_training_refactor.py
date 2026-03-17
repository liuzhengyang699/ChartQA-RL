import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import torch
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"Missing test dependency: {exc.name}")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "RL") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "RL"))

try:
    from verl.trainer.metrics import compute_structured_metrics
    from verl.trainer.ray_trainer import RayPPOTrainer
    from verl.utils.logger.logger import resolve_swanlab_log_dir
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"Missing runtime dependency: {exc.name}")


class DummyBatch:
    def __init__(self):
        self.batch = {
            "responses": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
        self.non_tensor_batch = {
            "metadata": np.array(
                [
                    json.dumps(
                        {
                            "type": "v_bar",
                            "x_values_bbox": {"A": [0, 0, 1, 1]},
                            "y_values_bbox": {},
                        }
                    )
                ],
                dtype=object,
            ),
            "query": np.array(["What is the value for A?"], dtype=object),
            "prompt": np.array(["prompt"], dtype=object),
        }

    def __len__(self):
        return len(self.non_tensor_batch["query"])


class StructuredMetricsTest(unittest.TestCase):
    def test_compute_structured_metrics_uses_consistent_rates(self):
        metrics = compute_structured_metrics(
            reward_metrics={
                "answer_accuracy": [1.0, 0.0, 0.5],
                "tool_gain": [0.4, 0.0, 0.2],
                "effective_tool": [1.0, 0.0, 0.0],
                "final_mix": [1.0, 0.2, 0.7],
                "baseline_mix": [0.6, 0.2, 0.5],
                "rule_score": [1.0, 0.0, 0.5],
                "judge_score": [1.0, 0.0, 1.0],
                "baseline_rule_score": [0.5, 0.0, 0.5],
                "baseline_judge_score": [0.5, 0.0, 0.5],
                "overall": [1.2, 0.2, 0.7],
            },
            action_valid=[True, True, False],
            tool_requested=[True, False, True],
            tool_exec_success=[True, False, False],
            invalid_action=[False, False, True],
            prefix="train",
        )

        self.assertAlmostEqual(metrics["train/ToolCallRate"], 2 / 3)
        self.assertAlmostEqual(metrics["train/LegalActionRate"], 2 / 3)
        self.assertAlmostEqual(metrics["train/ToolExecSuccessRate"], 1 / 1)
        self.assertAlmostEqual(metrics["train/AvgToolGain"], 0.4)
        self.assertAlmostEqual(metrics["train/ToolEffectivenessRate"], 1.0)
        self.assertAlmostEqual(metrics["train/InvalidActionRate"], 1 / 3)
        self.assertAlmostEqual(metrics["train/RewardScore"], (1.2 + 0.2 + 0.7) / 3)


class SwanlabLoggerTest(unittest.TestCase):
    def test_resolve_swanlab_log_dir_prefers_new_env_name(self):
        with mock.patch.dict(os.environ, {"SWANLAB_LOG_DIR": "/tmp/new", "SWANLAB_DIR": "/tmp/old"}, clear=True):
            self.assertEqual(resolve_swanlab_log_dir(), "/tmp/new")

    def test_resolve_swanlab_log_dir_falls_back_to_legacy_env_name(self):
        with mock.patch.dict(os.environ, {"SWANLAB_DIR": "/tmp/old"}, clear=True):
            self.assertEqual(resolve_swanlab_log_dir(), "/tmp/old")


class NoToolStructuredBatchTest(unittest.TestCase):
    def test_no_tool_mode_keeps_baseline_answer_and_valid_action(self):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.tokenizer = mock.Mock()
        trainer.tokenizer.decode.return_value = json.dumps(
            {
                "decision": "tool",
                "chart_axis": "x",
                "edit_mode": "highlight",
                "targets": ["A"],
            }
        )
        trainer.config = SimpleNamespace(algorithm=SimpleNamespace(enable_tool_branch=False))
        trainer._trace_enable = False
        trainer._trace_dir = None
        trainer._trace_max_steps = 0
        trainer.global_step = 1
        trainer._ensure_image_cache = mock.Mock()
        trainer._load_original_image = mock.Mock(return_value=Image.new("RGB", (8, 8), color="white"))
        trainer._generate_branch_outputs = mock.Mock(
            side_effect=[
                {0: {"text": "FINAL ANSWER: 42", "prompt_text": "baseline prompt"}},
                {},
            ]
        )

        batch = DummyBatch()
        processed_batch, stats = RayPPOTrainer._process_structured_chartqa_batch(trainer, batch)

        self.assertEqual(processed_batch.non_tensor_batch["baseline_answer_text"][0], "FINAL ANSWER: 42")
        self.assertEqual(processed_batch.non_tensor_batch["final_answer_text"][0], "FINAL ANSWER: 42")
        self.assertFalse(bool(processed_batch.non_tensor_batch["tool_exec_success"][0]))
        self.assertFalse(bool(processed_batch.non_tensor_batch["invalid_action"][0]))
        self.assertTrue(bool(processed_batch.non_tensor_batch["tool_requested"][0]))
        self.assertEqual(float(processed_batch.non_tensor_batch["tool_cost"][0]), 0.0)
        self.assertEqual(processed_batch.non_tensor_batch["tool_prompt_text"][0], "")
        self.assertEqual(processed_batch.non_tensor_batch["tool_answer_text"][0], "")
        self.assertEqual(stats["tool_requested"], 1)
        self.assertEqual(stats["tool_exec_success"], 0)
        self.assertEqual(stats["invalid_action"], 0)

        baseline_call = trainer._generate_branch_outputs.call_args_list[0].args[0]
        tool_call = trainer._generate_branch_outputs.call_args_list[1].args[0]
        self.assertEqual(len(baseline_call), 1)
        self.assertEqual(tool_call, [])


if __name__ == "__main__":
    unittest.main()
