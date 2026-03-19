import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.runtime import get_path_setting
from data.chartqa.rl import resolve_image_path, to_figure_path
from RL.verl.models.transformers.qwen3_vl import ensure_qwen3_vl_processor
from RL.verl.workers.actor import dp_actor
from RL.verl.workers.critic import dp_critic


class RuntimeConfigTest(unittest.TestCase):
    def test_rl_raw_dir_uses_supported_env_name(self):
        path_config = {"rl_raw_dir": Path("/configured/raw")}
        with mock.patch.dict(os.environ, {"CHARTQA_RL_RAW_DIR": "/env/raw"}, clear=True):
            self.assertEqual(get_path_setting(path_config, "rl_raw_dir"), Path("/env/raw").resolve())

    def test_rl_raw_dir_ignores_legacy_env_name(self):
        path_config = {"rl_raw_dir": Path("/configured/raw")}
        with mock.patch.dict(os.environ, {"CHARTQA_RAW_DIR": "/legacy/raw"}, clear=True):
            self.assertEqual(get_path_setting(path_config, "rl_raw_dir"), Path("/configured/raw").resolve())


class ChartQARLPathTest(unittest.TestCase):
    def test_to_figure_path_accepts_canonical_path(self):
        figure_path = "ChartQA/ChartQA Dataset/train/png/example.png"
        self.assertEqual(to_figure_path(figure_path), figure_path)

    def test_to_figure_path_rejects_legacy_prefix(self):
        with self.assertRaises(ValueError):
            to_figure_path("data/ChartQA/train/png/example.png")

    def test_resolve_image_path_uses_canonical_location_under_raw_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            image_path = raw_dir / "ChartQA" / "ChartQA Dataset" / "train" / "png" / "example.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.touch()

            resolved = resolve_image_path("ChartQA/ChartQA Dataset/train/png/example.png", raw_dir)

        self.assertEqual(resolved, image_path)


class FlashAttnImportTest(unittest.TestCase):
    def tearDown(self):
        dp_actor._get_flash_attn_padding_ops.cache_clear()
        dp_critic._get_flash_attn_padding_ops.cache_clear()

    def test_actor_padding_free_import_raises_clear_error_when_flash_attn_missing(self):
        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("flash_attn")):
            with self.assertRaisesRegex(RuntimeError, "padding_free=True requires flash-attn"):
                dp_actor._get_flash_attn_padding_ops()

    def test_critic_padding_free_import_raises_clear_error_when_flash_attn_missing(self):
        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("flash_attn")):
            with self.assertRaisesRegex(RuntimeError, "padding_free=True requires flash-attn"):
                dp_critic._get_flash_attn_padding_ops()


class Qwen3VLProcessorTest(unittest.TestCase):
    def test_accepts_fast_qwen_vl_image_processor(self):
        processor = type("Qwen3VLProcessor", (), {})()
        processor.image_processor = type("Qwen2VLImageProcessorFast", (), {})()

        self.assertIs(ensure_qwen3_vl_processor(processor), processor)


if __name__ == "__main__":
    unittest.main()
