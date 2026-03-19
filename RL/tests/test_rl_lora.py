import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "RL") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "RL"))

if torch is not None:
    from verl.rl_lora import (
        RL_LORA_CHECKPOINT_VERSION,
        build_rl_lora_checkpoint_metadata,
        create_adapter_state_dict,
        ensure_rl_lora_checkpoint_dir,
        iter_merged_weight_items,
        normalize_peft_weight_name,
        resolve_rl_lora_adapter_dir,
    )
else:  # pragma: no cover - environment dependent
    RL_LORA_CHECKPOINT_VERSION = "rl_lora_v1"


@unittest.skipIf(torch is None, "Missing test dependency: torch")
class RLLoRAHelpersTest(unittest.TestCase):
    def test_normalize_peft_weight_name_restores_base_model_keys(self):
        self.assertEqual(
            normalize_peft_weight_name("base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"),
            "model.layers.0.self_attn.q_proj.weight",
        )

    def test_create_adapter_state_dict_keeps_only_adapter_weights(self):
        adapter_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        adapter_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
        state_dict = {
            "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.ones(2, 2),
            adapter_a: torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            adapter_b: torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        }

        adapter_state = create_adapter_state_dict(state_dict)

        self.assertEqual(set(adapter_state), {adapter_a, adapter_b})
        self.assertTrue(torch.equal(adapter_state[adapter_a], state_dict[adapter_a]))

    def test_iter_merged_weight_items_merges_lora_delta_into_base_weights(self):
        state_dict = {
            "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]]
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.tensor(
                [[2.0, 0.0], [0.0, 2.0]]
            ),
            "base_model.model.lm_head.weight": torch.tensor([[5.0, 6.0]]),
        }
        rl_lora_config = SimpleNamespace(enable=True, r=2, alpha=4, dropout=0.05, target_modules=("q_proj",))

        merged = dict(iter_merged_weight_items(state_dict, rl_lora_config))

        expected_delta = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
        self.assertTrue(
            torch.equal(
                merged["model.layers.0.self_attn.q_proj.weight"],
                state_dict["base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"] + expected_delta,
            )
        )
        self.assertTrue(torch.equal(merged["model.lm_head.weight"], state_dict["base_model.model.lm_head.weight"]))


@unittest.skipIf(torch is None, "Missing test dependency: torch")
class RLLoRACheckpointFormatTest(unittest.TestCase):
    def test_checkpoint_dir_requires_new_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "adapter-first checkpoint"):
                ensure_rl_lora_checkpoint_dir(tmpdir)

    def test_resolve_adapter_dir_accepts_actor_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            adapter_dir = checkpoint_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            metadata = build_rl_lora_checkpoint_metadata("/tmp/base_model", SimpleNamespace(enable=True, r=8, alpha=16, dropout=0.05, target_modules=("q_proj", "k_proj")))
            (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (adapter_dir / "adapter_model.bin").write_bytes(b"adapter")
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/tmp/base_model", "peft_type": "LORA"}),
                encoding="utf-8",
            )

            resolved = resolve_rl_lora_adapter_dir(checkpoint_dir)

        self.assertEqual(resolved.name, "adapter")

    def test_build_checkpoint_metadata_marks_new_version(self):
        metadata = build_rl_lora_checkpoint_metadata(
            "/tmp/base_model",
            SimpleNamespace(enable=True, r=8, alpha=16, dropout=0.05, target_modules=("q_proj", "v_proj")),
        )

        self.assertEqual(metadata["format_version"], RL_LORA_CHECKPOINT_VERSION)
        self.assertEqual(metadata["base_model_path"], "/tmp/base_model")
        self.assertEqual(metadata["rl_lora"]["target_modules"], ["q_proj", "v_proj"])


if __name__ == "__main__":
    unittest.main()
