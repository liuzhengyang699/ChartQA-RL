# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import importlib.util
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import json

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .metrics import (
    compute_data_metrics,
    compute_structured_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)

from ..tooluse.structured_chartqa import (
    build_baseline_answer_prompt,
    build_generation_feature,
    build_tool_answer_prompt,
    build_supervised_feature,
    execute_validated_action,
    parse_action_response,
    validate_action_payload,
)
from PIL import Image
from ..utils.dataset import collate_fn
from .replay_buffer import ReplayBuffer

import sys
import traceback


def _load_rl_raw_dir(repo_root: Path) -> Optional[str]:
    for env_var in ("CHARTQA_RL_RAW_DIR", "CHARTQA_RAW_DIR"):
        override = os.getenv(env_var)
        if override:
            return str(Path(override).expanduser().resolve())

    config_module_path = repo_root / "LoRA" / "utils" / "config.py"
    if not config_module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("chartqa_lora_config", config_module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        path_config, _ = module.load_path_config()
        return str(module.get_path_setting(path_config, "rl_raw_dir"))
    except Exception:
        return None


def _figure_path_variants(figure_path: str) -> List[str]:
    normalized = figure_path.replace("\\", "/").strip()
    variants: List[str] = []

    def add(value: str) -> None:
        if value and value not in variants:
            variants.append(value)

    add(normalized)
    if normalized.startswith("data/ChartQA/"):
        suffix = normalized[len("data/ChartQA/") :]
        add(f"ChartQA/ChartQA Dataset/{suffix}")
    if "ChartQA/ChartQA Dataset/" in normalized:
        suffix = normalized.split("ChartQA/ChartQA Dataset/", 1)[1]
        add(f"ChartQA/ChartQA Dataset/{suffix}")
    for split in ("train", "val", "test"):
        split_prefix = f"{split}/"
        if normalized.startswith(split_prefix):
            add(f"ChartQA/ChartQA Dataset/{normalized}")
            break
    return variants

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    _set_batch_tensor(data, "token_level_rewards", token_level_scores - kl_ctrl.kl_coef * kld)

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def _set_batch_tensor(data: DataProto, key: str, value: torch.Tensor) -> None:
    try:
        if key in data.batch.keys():
            data.batch.set_(key, value)
        else:
            data.batch = data.batch.clone()
            data.batch.set(key, value)
    except RuntimeError:
        data.batch = data.batch.clone()
        data.batch.set(key, value)


def _build_experiment_metrics(config: PPOConfig) -> Dict[str, float]:
    return {
        "experiment/disable_kl": float(bool(config.algorithm.disable_kl)),
        "experiment/use_kl_loss": float(bool(config.algorithm.use_kl_loss)),
        "experiment/enable_tool_branch": float(bool(config.algorithm.enable_tool_branch)),
    }

def custom_unraisablehook(unraisable):
    print("=== Unraisable Exception Caught ===")
    print(f"Type: {unraisable.exc_type.__name__}")
    print(f"Value: {unraisable.exc_value}")
    if unraisable.object:
        print(f"In object: {unraisable.object}")
    if unraisable.traceback:
        print("Traceback:")
        traceback.print_tb(unraisable.traceback)
    print("===================================")

class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.logger: Optional[Tracker] = None
        self._experiment_metrics = _build_experiment_metrics(config)

        sys.unraisablehook = custom_unraisablehook

        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self._rl_raw_dir = _load_rl_raw_dir(Path(self._project_root))

        self._trace_enable = bool(getattr(config.trainer, "trace", None) and config.trainer.trace.enable)
        self._trace_dir = None
        self._trace_sample_size = 0
        self._trace_save_images = False
        self._trace_max_steps = 0
        if getattr(config.trainer, "trace", None) is not None:
            self._trace_dir = config.trainer.trace.output_dir
            self._trace_sample_size = int(config.trainer.trace.sample_size_per_step)
            self._trace_save_images = bool(config.trainer.trace.save_images)
            self._trace_max_steps = int(config.trainer.trace.max_steps)
        if self._trace_enable and self._trace_dir:
            os.makedirs(self._trace_dir, exist_ok=True)

        self.replay_buffer: Optional[ReplayBuffer] = None
        replay_config = getattr(config.trainer, "replay", None)
        if replay_config is not None and replay_config.enable:
            self.replay_buffer = ReplayBuffer(
                buffer_dir=replay_config.buffer_dir,
                buffer_size=replay_config.buffer_size,
                per_figure_limit=3,
                min_final_mix=replay_config.min_final_mix,
                min_tool_gain=replay_config.min_tool_gain,
                seed=getattr(config.data, "seed", 42),
            )

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")
        if config.algorithm.action_mode != "structured":
            raise NotImplementedError(
                "This ChartQA fork only supports structured action mode. "
                "Legacy free-form tool-use paths have been removed."
            )

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def _structured_metrics_for_batch(self, batch: DataProto, reward_metrics: Dict[str, List[float]], prefix: str) -> Dict[str, float]:
        return compute_structured_metrics(
            reward_metrics=reward_metrics,
            action_valid=batch.non_tensor_batch["action_valid"],
            tool_requested=batch.non_tensor_batch["tool_requested"],
            tool_exec_success=batch.non_tensor_batch["tool_exec_success"],
            invalid_action=batch.non_tensor_batch["invalid_action"],
            prefix=prefix,
        )

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _maybe_trace_step(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_metrics: Dict[str, List[float]],
    ) -> None:
        if not self._trace_enable or not self._trace_dir:
            return
        if self._trace_max_steps > 0 and self.global_step > self._trace_max_steps:
            return
        if self._trace_sample_size <= 0:
            return

        step_dir = os.path.join(self._trace_dir, f"step_{self.global_step:06d}")
        os.makedirs(step_dir, exist_ok=True)

        response_ids = batch.batch.get("responses", None)
        response_mask = batch.batch.get("response_mask", None)
        reward_scalar = reward_tensor.sum(dim=-1).detach().float().cpu().tolist()

        prompts = batch.non_tensor_batch.get("prompt", None)
        queries = batch.non_tensor_batch.get("query", None)
        gts = batch.non_tensor_batch.get("ground_truth", None)
        penalties = batch.non_tensor_batch.get("penalty", None)
        uids = batch.non_tensor_batch.get("uid", None)
        figure_ids = batch.non_tensor_batch.get("figure_id", None)
        figure_paths = batch.non_tensor_batch.get("figure_path", None)
        image_1_pil = batch.non_tensor_batch.get("image_1_pil", None)
        multi_modal_data = batch.non_tensor_batch.get("multi_modal_data", None)
        rollout_rounds = batch.non_tensor_batch.get("rollout_round", None)
        tool_codes = batch.non_tensor_batch.get("tool_code", None)
        tool_parse_status = batch.non_tensor_batch.get("tool_parse_status", None)
        tool_error_code = batch.non_tensor_batch.get("tool_error_code", None)
        tool_exec_success = batch.non_tensor_batch.get("tool_exec_success", None)
        tool_second_prompt = batch.non_tensor_batch.get("tool_second_prompt", None)
        tool_edited_image_path = batch.non_tensor_batch.get("tool_edited_image_path", None)
        tool_original_image_path = batch.non_tensor_batch.get("tool_original_image_path", None)

        indices: List[int] = []
        seen = set()
        for i in range(len(batch)):
            uid = str(uids[i]) if uids is not None else str(i)
            if uid in seen:
                continue
            seen.add(uid)
            indices.append(i)
            if len(indices) >= self._trace_sample_size:
                break

        trace_path = os.path.join(step_dir, "trace.jsonl")
        with open(trace_path, "w", encoding="utf-8") as f:
            for i in indices:
                resp_text = None
                if response_ids is not None and response_mask is not None:
                    resp_len = int(response_mask[i].sum().item())
                    resp_text = self.tokenizer.decode(response_ids[i][:resp_len], skip_special_tokens=True)

                record: Dict[str, Any] = {
                    "global_step": int(self.global_step),
                    "index": int(i),
                    "uid": str(uids[i]) if uids is not None else None,
                    "rollout_round": int(rollout_rounds[i]) if rollout_rounds is not None else 0,
                    "figure_id": str(figure_ids[i]) if figure_ids is not None else None,
                    "figure_path": str(figure_paths[i]) if figure_paths is not None else None,
                    "query": str(queries[i]) if queries is not None else None,
                    "ground_truth": str(gts[i]) if gts is not None else None,
                    "prompt": str(prompts[i]) if prompts is not None else None,
                    "response": resp_text,
                    "penalty": float(penalties[i]) if penalties is not None else 0.0,
                    "reward_overall": float(reward_scalar[i]),
                }
                for k, v in reward_metrics.items():
                    if i < len(v):
                        record[f"reward_{k}"] = float(v[i])
                record["tool_parse_status"] = int(tool_parse_status[i]) if tool_parse_status is not None else None
                record["tool_error_code"] = str(tool_error_code[i]) if tool_error_code is not None else None
                record["tool_code"] = str(tool_codes[i]) if tool_codes is not None else None
                record["tool_exec_success"] = int(tool_exec_success[i]) if tool_exec_success is not None else None
                record["tool_second_prompt"] = str(tool_second_prompt[i]) if tool_second_prompt is not None else None
                record["tool_edited_image_path"] = (
                    str(tool_edited_image_path[i]) if tool_edited_image_path is not None else None
                )
                record["tool_original_image_path"] = (
                    str(tool_original_image_path[i]) if tool_original_image_path is not None else None
                )

                if self._trace_save_images:
                    saved_original = None
                    if tool_original_image_path is not None and i < len(tool_original_image_path) and tool_original_image_path[i]:
                        record["saved_original_image"] = str(tool_original_image_path[i])
                        saved_original = str(tool_original_image_path[i])
                    if figure_paths is not None and figure_paths[i]:
                        try:
                            img = Image.open(str(figure_paths[i]))
                            saved_original = os.path.join(step_dir, f"img_{i}_orig.png")
                            img.save(saved_original)
                        except Exception:
                            saved_original = None
                    if saved_original is None and image_1_pil is not None and i < len(image_1_pil):
                        try:
                            if isinstance(image_1_pil[i], Image.Image):
                                saved_original = os.path.join(step_dir, f"img_{i}_orig.png")
                                image_1_pil[i].save(saved_original)
                        except Exception:
                            saved_original = None
                    if saved_original is None and multi_modal_data is not None and i < len(multi_modal_data):
                        try:
                            mm = multi_modal_data[i]
                            if isinstance(mm, dict):
                                imgs = mm.get("image", None)
                                if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                                    saved_original = os.path.join(step_dir, f"img_{i}_orig.png")
                                    imgs[0].save(saved_original)
                        except Exception:
                            saved_original = None
                    if saved_original is not None:
                        record["saved_original_image"] = saved_original

                    if tool_edited_image_path is not None and tool_edited_image_path[i]:
                        record["saved_edited_image"] = str(tool_edited_image_path[i])

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _ensure_image_cache(self, batch: DataProto) -> None:
        if "image_1_pil" in batch.non_tensor_batch:
            return
        mm = batch.non_tensor_batch.get("multi_modal_data", None)
        if mm is None:
            return
        image_1_pil = []
        for item in mm:
            img0 = None
            if isinstance(item, dict):
                imgs = item.get("image", None)
                if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                    img0 = imgs[0]
            image_1_pil.append(img0)
        batch.non_tensor_batch["image_1_pil"] = np.array(image_1_pil, dtype=object)

    def _resolve_figure_path(self, figure_path: str) -> str:
        candidates: List[str] = []
        if os.path.isabs(figure_path):
            candidates.append(figure_path)
        else:
            for variant in _figure_path_variants(figure_path):
                if self._rl_raw_dir:
                    candidates.append(os.path.join(self._rl_raw_dir, variant))
                    if variant.startswith("data/"):
                        candidates.append(os.path.join(self._rl_raw_dir, variant[len("data/") :]))
                candidates.extend(
                    [
                        variant,
                        os.path.join(os.getcwd(), variant),
                        os.path.join(self._project_root, variant),
                    ]
                )
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(figure_path)

    def _load_original_image(self, batch: DataProto, index: int) -> Image.Image:
        image_1_pil = batch.non_tensor_batch.get("image_1_pil", None)
        if image_1_pil is not None and index < len(image_1_pil) and isinstance(image_1_pil[index], Image.Image):
            return image_1_pil[index].copy()

        figure_path = str(batch.non_tensor_batch["figure_path"][index])
        with Image.open(self._resolve_figure_path(figure_path)) as image:
            return image.convert("RGB")

    def _generate_branch_outputs(self, requests: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        if not requests:
            return {}

        process_image = self.val_dataloader.dataset.process_image
        features = []
        for request in requests:
            feature = build_generation_feature(
                processor=self.processor,
                tokenizer=self.tokenizer,
                prompt_text=request["prompt_text"],
                images=request["images"],
                max_prompt_length=self.val_dataloader.dataset.max_prompt_length,
                truncation=self.val_dataloader.dataset.truncation,
                process_image=process_image,
            )
            feature["request_index"] = request["request_index"]
            feature["prompt_text"] = request["prompt_text"]
            features.append(feature)

        branch_batch = DataProto.from_single_dict(collate_fn(features))
        gen_batch = branch_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
        )
        gen_batch.meta_info = dict(self.config.worker.rollout.val_override_config)
        gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        output_batch = self.actor_rollout_wg.generate_sequences(gen_batch)
        output_batch = unpad_dataproto(output_batch, pad_size=pad_size)
        branch_batch = branch_batch.union(output_batch)

        response_ids = branch_batch.batch["responses"]
        results: Dict[int, Dict[str, Any]] = {}
        for i in range(len(branch_batch)):
            decoded = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)
            results[int(branch_batch.non_tensor_batch["request_index"][i])] = {
                "text": decoded,
                "prompt_text": str(branch_batch.non_tensor_batch["prompt_text"][i]),
            }
        return results

    def _process_structured_chartqa_batch(self, batch: DataProto) -> tuple[DataProto, Dict[str, int]]:
        self._ensure_image_cache(batch)
        enable_tool_branch = bool(self.config.algorithm.enable_tool_branch)
        response_ids = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        response_texts = []
        for i in range(len(batch)):
            resp_len = int(response_mask[i].sum().item())
            response_texts.append(self.tokenizer.decode(response_ids[i][:resp_len], skip_special_tokens=True))

        n = len(batch)
        action_valid = np.zeros(n, dtype=object)
        invalid_action = np.zeros(n, dtype=object)
        tool_requested = np.zeros(n, dtype=object)
        tool_exec_success = np.zeros(n, dtype=object)
        decisions = np.empty(n, dtype=object)
        chart_axes = np.empty(n, dtype=object)
        edit_modes = np.empty(n, dtype=object)
        targets_json = np.empty(n, dtype=object)
        tool_names = np.empty(n, dtype=object)
        tool_costs = np.zeros(n, dtype=object)
        action_target_json = np.empty(n, dtype=object)
        action_error_code = np.empty(n, dtype=object)
        baseline_prompt_text = np.empty(n, dtype=object)
        tool_prompt_text = np.empty(n, dtype=object)
        answer_prompt_text = np.empty(n, dtype=object)
        baseline_answer_text = np.empty(n, dtype=object)
        tool_answer_text = np.empty(n, dtype=object)
        final_answer_text = np.empty(n, dtype=object)
        tool_edited_image_path = np.empty(n, dtype=object)
        tool_original_image_path = np.empty(n, dtype=object)

        baseline_requests: List[Dict[str, Any]] = []
        tool_requests: List[Dict[str, Any]] = []

        trace_enable = self._trace_enable and bool(self._trace_dir) and (self._trace_max_steps <= 0 or self.global_step <= self._trace_max_steps)
        trace_step_dir = None
        if trace_enable:
            trace_step_dir = os.path.join(self._trace_dir, f"step_{self.global_step:06d}")
            os.makedirs(trace_step_dir, exist_ok=True)

        stats = {
            "tool_requested": 0,
            "action_legal": 0,
            "tool_legal": 0,
            "tool_exec_success": 0,
            "invalid_action": 0,
        }

        for index, response_text in enumerate(response_texts):
            metadata = json.loads(batch.non_tensor_batch["metadata"][index])
            query = str(batch.non_tensor_batch["query"][index])
            parse_result = parse_action_response(response_text)
            validated = validate_action_payload(parse_result.get("payload"), metadata)
            decisions[index] = validated["decision"]
            chart_axes[index] = validated["chart_axis"]
            edit_modes[index] = validated["edit_mode"]
            targets_json[index] = json.dumps(validated["targets"], ensure_ascii=False)
            tool_names[index] = validated.get("tool_name", "")
            action_target_json[index] = validated["canonical_action_json"]
            action_error_code[index] = validated.get("error_code") or parse_result.get("error_code") or ""

            tool_requested[index] = validated["decision"] == "tool"
            if tool_requested[index]:
                stats["tool_requested"] += 1

            original_image = self._load_original_image(batch, index)
            if trace_enable and trace_step_dir is not None:
                original_path = os.path.join(trace_step_dir, f"img_{index}_orig.png")
                original_image.save(original_path)
                tool_original_image_path[index] = original_path
            else:
                tool_original_image_path[index] = ""

            baseline_prompt = build_baseline_answer_prompt(query)
            baseline_prompt_text[index] = baseline_prompt
            baseline_requests.append(
                {
                    "request_index": index,
                    "prompt_text": baseline_prompt,
                    "images": [original_image.copy()],
                }
            )

            action_valid[index] = bool(validated["valid"])
            if action_valid[index]:
                stats["action_legal"] += 1
            if action_valid[index] and tool_requested[index]:
                stats["tool_legal"] += 1

            executed = {
                **validated,
                "tool_executed": False,
                "tool_exec_success": False,
                "tool_error_code": "",
                "edited_image": None,
            }
            if enable_tool_branch and validated["decision"] == "tool" and validated["valid"]:
                executed = execute_validated_action(validated, original_image.copy(), metadata)

            if enable_tool_branch and validated["decision"] == "tool" and validated["valid"]:
                tool_costs[index] = 0.05 + 0.01 * max(0, len(validated["targets"]) - 1)
            else:
                tool_costs[index] = 0.0

            if enable_tool_branch and executed["decision"] == "tool" and executed["tool_exec_success"]:
                tool_exec_success[index] = True
                stats["tool_exec_success"] += 1
                tool_prompt = build_tool_answer_prompt(query, executed)
                tool_prompt_text[index] = tool_prompt
                tool_requests.append(
                    {
                        "request_index": index,
                        "prompt_text": tool_prompt,
                        "images": [original_image.copy(), executed["edited_image"].copy()],
                    }
                )
                if trace_enable and trace_step_dir is not None:
                    edited_path = os.path.join(trace_step_dir, f"img_{index}_edited.png")
                    executed["edited_image"].save(edited_path)
                    tool_edited_image_path[index] = edited_path
                else:
                    tool_edited_image_path[index] = ""
            else:
                tool_exec_success[index] = False
                tool_prompt_text[index] = ""
                tool_edited_image_path[index] = ""

            invalid_flag = (not validated["valid"]) or (
                enable_tool_branch and validated["decision"] == "tool" and not executed["tool_exec_success"]
            )
            invalid_action[index] = bool(invalid_flag)
            if invalid_flag:
                stats["invalid_action"] += 1

            answer_prompt_text[index] = baseline_prompt
            baseline_answer_text[index] = ""
            tool_answer_text[index] = ""
            final_answer_text[index] = ""

        baseline_outputs = self._generate_branch_outputs(baseline_requests)
        tool_outputs = self._generate_branch_outputs(tool_requests)

        for index in range(n):
            baseline_answer_text[index] = baseline_outputs.get(index, {}).get("text", "")
            final_answer_text[index] = baseline_answer_text[index]
            answer_prompt_text[index] = baseline_outputs.get(index, {}).get("prompt_text", str(baseline_prompt_text[index]))
            if bool(tool_exec_success[index]):
                tool_answer_text[index] = tool_outputs.get(index, {}).get("text", "")
                if tool_answer_text[index]:
                    final_answer_text[index] = tool_answer_text[index]
                    answer_prompt_text[index] = tool_outputs.get(index, {}).get("prompt_text", str(tool_prompt_text[index]))

        batch.non_tensor_batch["action_valid"] = action_valid
        batch.non_tensor_batch["invalid_action"] = invalid_action
        batch.non_tensor_batch["tool_requested"] = tool_requested
        batch.non_tensor_batch["tool_exec_success"] = tool_exec_success
        batch.non_tensor_batch["decision"] = decisions
        batch.non_tensor_batch["chart_axis"] = chart_axes
        batch.non_tensor_batch["edit_mode"] = edit_modes
        batch.non_tensor_batch["targets_json"] = targets_json
        batch.non_tensor_batch["tool_name"] = tool_names
        batch.non_tensor_batch["tool_cost"] = tool_costs
        batch.non_tensor_batch["action_target_json"] = action_target_json
        batch.non_tensor_batch["action_error_code"] = action_error_code
        batch.non_tensor_batch["baseline_prompt_text"] = baseline_prompt_text
        batch.non_tensor_batch["tool_prompt_text"] = tool_prompt_text
        batch.non_tensor_batch["answer_prompt_text"] = answer_prompt_text
        batch.non_tensor_batch["baseline_answer_text"] = baseline_answer_text
        batch.non_tensor_batch["tool_answer_text"] = tool_answer_text
        batch.non_tensor_batch["final_answer_text"] = final_answer_text
        batch.non_tensor_batch["tool_edited_image_path"] = tool_edited_image_path
        batch.non_tensor_batch["tool_original_image_path"] = tool_original_image_path
        return batch, stats

    def _build_replay_entries(self, batch: DataProto, reward_metrics: Dict[str, List[float]]) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        image_dir = None
        if self.replay_buffer is not None:
            image_dir = self.replay_buffer.buffer_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(batch)):
            final_answer = str(batch.non_tensor_batch["final_answer_text"][i])
            baseline_answer = str(batch.non_tensor_batch["baseline_answer_text"][i])
            if not final_answer or not baseline_answer:
                continue

            tool_metadata = {
                "decision": str(batch.non_tensor_batch["decision"][i]),
                "chart_axis": str(batch.non_tensor_batch["chart_axis"][i]),
                "edit_mode": str(batch.non_tensor_batch["edit_mode"][i]),
                "targets_json": str(batch.non_tensor_batch["targets_json"][i]),
                "tool_name": str(batch.non_tensor_batch["tool_name"][i]),
                "metadata_json": str(batch.non_tensor_batch["metadata"][i]),
                "tool_exec_success": bool(batch.non_tensor_batch["tool_exec_success"][i]),
            }
            final_mix = float(reward_metrics["final_mix"][i])
            baseline_mix = float(reward_metrics["baseline_mix"][i])
            tool_gain = float(reward_metrics["tool_gain"][i])
            quality_score = final_mix + max(0.0, tool_gain)
            decision = tool_metadata["decision"]
            effective_tool = float(reward_metrics["effective_tool"][i]) > 0.5
            invalid_flag = bool(batch.non_tensor_batch["invalid_action"][i])
            stored_image_path = ""
            if image_dir is not None:
                image_name = f"{str(batch.non_tensor_batch['figure_id'][i]).replace('/', '_')}.png"
                stored_path = image_dir / image_name
                if not stored_path.exists():
                    original_image = batch.non_tensor_batch.get("image_1_pil", [None] * len(batch))[i]
                    if isinstance(original_image, Image.Image):
                        original_image.save(stored_path)
                if stored_path.exists():
                    stored_image_path = str(stored_path)

            if effective_tool:
                bucket = "tool_positive"
                action_json = str(batch.non_tensor_batch["action_target_json"][i])
                answer_prompt = str(batch.non_tensor_batch["answer_prompt_text"][i])
                answer_target = final_answer
                entry_final_mix = final_mix
                entry_tool_gain = tool_gain
            elif decision == "direct" and final_mix >= 0.9:
                bucket = "direct_high_confidence"
                action_json = str(batch.non_tensor_batch["action_target_json"][i])
                answer_prompt = str(batch.non_tensor_batch["answer_prompt_text"][i])
                answer_target = final_answer
                entry_final_mix = final_mix
                entry_tool_gain = tool_gain
            elif (invalid_flag or tool_gain <= 0.0) and baseline_mix >= 0.9:
                bucket = "hard_negative_repaired"
                repaired_tool_metadata = {
                    **tool_metadata,
                    "decision": "direct",
                    "chart_axis": "x",
                    "edit_mode": "highlight",
                    "targets_json": "[]",
                    "repaired_from_invalid": invalid_flag,
                }
                entries.append(
                    {
                        "figure_id": str(batch.non_tensor_batch["figure_id"][i]),
                        "figure_path": str(batch.non_tensor_batch["figure_path"][i]),
                        "query": str(batch.non_tensor_batch["query"][i]),
                        "action_prompt": str(batch.non_tensor_batch["formatted_prompt_text"][i]),
                        "action_target_json": json.dumps(
                            {"decision": "direct", "chart_axis": "x", "edit_mode": "highlight", "targets": []},
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "answer_prompt": str(batch.non_tensor_batch["baseline_prompt_text"][i]),
                        "answer_target_text": baseline_answer,
                        "quality_score": baseline_mix,
                        "tool_gain": 0.0,
                        "final_mix": baseline_mix,
                        "baseline_mix": baseline_mix,
                        "bucket": bucket,
                        "stored_image_path": stored_image_path,
                        "tool_metadata": repaired_tool_metadata,
                    }
                )
                continue
            else:
                continue

            entries.append(
                {
                    "figure_id": str(batch.non_tensor_batch["figure_id"][i]),
                    "figure_path": str(batch.non_tensor_batch["figure_path"][i]),
                    "query": str(batch.non_tensor_batch["query"][i]),
                    "action_prompt": str(batch.non_tensor_batch["formatted_prompt_text"][i]),
                    "action_target_json": action_json,
                    "answer_prompt": answer_prompt,
                    "answer_target_text": answer_target,
                    "quality_score": quality_score,
                    "tool_gain": entry_tool_gain,
                    "final_mix": entry_final_mix,
                    "baseline_mix": baseline_mix,
                    "bucket": bucket,
                    "stored_image_path": stored_image_path,
                    "tool_metadata": tool_metadata,
                }
            )
        return entries

    def _maybe_update_replay_buffer(self, batch: DataProto, reward_metrics: Dict[str, List[float]]) -> None:
        if self.replay_buffer is None:
            return
        entries = self._build_replay_entries(batch, reward_metrics)
        self.replay_buffer.add_entries(entries)

    def _reconstruct_answer_images(self, entry: Dict[str, object]) -> List[Image.Image]:
        source_path = str(entry.get("stored_image_path") or entry["figure_path"])
        figure_path = self._resolve_figure_path(source_path)
        with Image.open(figure_path) as image:
            original_image = image.convert("RGB")

        tool_metadata = dict(entry.get("tool_metadata", {}))
        decision = str(tool_metadata.get("decision", "direct"))
        if decision != "tool" or not bool(tool_metadata.get("tool_exec_success", False)):
            return [original_image]

        metadata = json.loads(str(tool_metadata["metadata_json"]))
        action = {
            "decision": "tool",
            "chart_axis": str(tool_metadata.get("chart_axis", "x")),
            "edit_mode": str(tool_metadata.get("edit_mode", "highlight")),
            "targets": json.loads(str(tool_metadata.get("targets_json", "[]"))),
        }
        validated = validate_action_payload(action, metadata)
        executed = execute_validated_action(validated, original_image.copy(), metadata)
        if executed["tool_exec_success"]:
            return [original_image, executed["edited_image"]]
        return [original_image]

    def _build_replay_supervised_batch(
        self,
        action_entries: List[Dict[str, object]],
        answer_entries: List[Dict[str, object]],
    ) -> Optional[DataProto]:
        if not action_entries and not answer_entries:
            return None

        process_image = self.train_dataloader.dataset.process_image
        features: List[Dict[str, Any]] = []
        replay_config = self.config.trainer.replay

        for entry in action_entries:
            figure_path = self._resolve_figure_path(str(entry["figure_path"]))
            with Image.open(figure_path) as image:
                original_image = image.convert("RGB")
            feature = build_supervised_feature(
                processor=self.processor,
                tokenizer=self.tokenizer,
                prompt_text=str(entry["action_prompt"]),
                images=[original_image],
                assistant_text=str(entry["action_target_json"]),
                max_prompt_length=self.train_dataloader.dataset.max_prompt_length,
                truncation=self.train_dataloader.dataset.truncation,
                process_image=process_image,
                loss_weight=replay_config.loss_weight_action,
            )
            feature["supervision_kind"] = "action"
            features.append(feature)

        for entry in answer_entries:
            images = self._reconstruct_answer_images(entry)
            feature = build_supervised_feature(
                processor=self.processor,
                tokenizer=self.tokenizer,
                prompt_text=str(entry["answer_prompt"]),
                images=images,
                assistant_text=str(entry["answer_target_text"]),
                max_prompt_length=self.train_dataloader.dataset.max_prompt_length,
                truncation=self.train_dataloader.dataset.truncation,
                process_image=process_image,
                loss_weight=replay_config.loss_weight_answer,
            )
            feature["supervision_kind"] = "answer"
            features.append(feature)

        if not features:
            return None
        replay_batch = DataProto.from_single_dict(collate_fn(features))
        replay_batch.meta_info["global_token_num"] = torch.sum(replay_batch.batch["attention_mask"], dim=-1).tolist()
        return replay_batch

    def _maybe_run_replay_update(self, metrics: Dict[str, Any]) -> None:
        if self.replay_buffer is None or len(self.replay_buffer) == 0:
            return

        replay_config = self.config.trainer.replay
        action_entries, answer_entries = self.replay_buffer.sample_supervision(
            total_batch_size=replay_config.supervised_batch_size,
            action_weight=replay_config.loss_weight_action,
            answer_weight=replay_config.loss_weight_answer,
        )
        replay_batch = self._build_replay_supervised_batch(action_entries, answer_entries)
        if replay_batch is None:
            return

        actor_output = self.actor_rollout_wg.update_actor_supervised(replay_batch)
        aux_metrics = reduce_metrics(actor_output.non_tensor_batch)
        metrics.update(aux_metrics)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        action_valid_all: List[bool] = []
        tool_requested_all: List[bool] = []
        tool_exec_success_all: List[bool] = []
        invalid_action_all: List[bool] = []

        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            self._ensure_image_cache(test_batch)

            sample_inputs.extend(test_batch.non_tensor_batch["query"].tolist())
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())

            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch, _ = self._process_structured_chartqa_batch(test_batch)

            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))
            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            sample_outputs.extend(test_batch.non_tensor_batch["final_answer_text"].tolist())
            sample_scores.extend(reward_tensor.sum(-1).cpu().tolist())
            action_valid_all.extend(test_batch.non_tensor_batch["action_valid"].astype(bool).tolist())
            tool_requested_all.extend(test_batch.non_tensor_batch["tool_requested"].astype(bool).tolist())
            tool_exec_success_all.extend(test_batch.non_tensor_batch["tool_exec_success"].astype(bool).tolist())
            invalid_action_all.extend(test_batch.non_tensor_batch["invalid_action"].astype(bool).tolist())

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item() if reward_tensor_lst else 0.0
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        structured_metrics = compute_structured_metrics(
            reward_metrics=reward_metrics_lst,
            action_valid=action_valid_all,
            tool_requested=tool_requested_all,
            tool_exec_success=tool_exec_success_all,
            invalid_action=invalid_action_all,
            prefix="val",
        )
        return {
            "val/reward_score": reward_score,
            **structured_metrics,
            **val_reward_metrics,
            **self._experiment_metrics,
        }

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        try:
            # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
            remove_obsolete_ckpt(
                self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
            )
            folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
            actor_path = os.path.join(folder_path, "actor")
            self.actor_rollout_wg.save_checkpoint(actor_path)

            if self.use_critic:
                critic_path = os.path.join(folder_path, "critic")
                self.critic_wg.save_checkpoint(critic_path)

            dataloader_path = os.path.join(folder_path, "dataloader.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_path)

            last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
            with open(last_global_step_path, "w") as f:
                f.write(str(self.global_step))
        except Exception as e:
            print("ERROR")
            print(e)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size

        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data_online_filtering(self, metrics: Dict[str, Any]) -> DataProto:
        batch = None
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            new_batch: DataProto = DataProto.from_single_dict(batch_dict)
            new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch))], dtype=object)

            if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                mm = new_batch.non_tensor_batch.get("multi_modal_data", None)
                if mm is not None and "image_1_pil" not in new_batch.non_tensor_batch:
                    image_1_pil = []
                    for item in mm:
                        img0 = None
                        if isinstance(item, dict):
                            imgs = item.get("image", None)
                            if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                                img0 = imgs[0]
                        image_1_pil.append(img0)
                    new_batch.non_tensor_batch["image_1_pil"] = np.array(image_1_pil, dtype=object)
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
            else:
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                _set_batch_tensor(new_batch, "reward_baselines", reward_baseline_tensor)
                del gen_baseline_batch, gen_baseline_output

            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)
            if "metadata" not in new_batch.non_tensor_batch:
                raise KeyError("Structured ChartQA training expects metadata in the RL batch.")
            new_batch, _ = self._process_structured_chartqa_batch(new_batch)
            new_batch.non_tensor_batch.pop("multi_modal_data", None)

            reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            _set_batch_tensor(new_batch, "token_level_scores", reward_tensor)

            filter_scores = reward_metrics.get(self.config.algorithm.filter_key, None)
            if filter_scores is None:
                raise KeyError(f"Missing filter_key={self.config.algorithm.filter_key} in reward_metrics.")

            uids = new_batch.non_tensor_batch["uid"]
            uid2scores = defaultdict(list)
            for uid, score in zip(uids, filter_scores):
                uid2scores[uid].append(score)

            uid2mean = {uid: float(np.mean(scores)) for uid, scores in uid2scores.items()}
            kept_uids = [
                uid
                for uid, avg_score in uid2mean.items()
                if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
            ]
            kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
            if len(kept_sample_idxs) == 0:
                raise RuntimeError("No sample is kept after filtering. Please check your data.")

            new_batch.reorder(torch.tensor(kept_sample_idxs, dtype=torch.long))

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = getattr(self.config.trainer, "max_try_make_batch", 20)
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data.")
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        try:
            self.global_step = 0
            val_metrics: Optional[Dict[str, Any]] = None

            # load checkpoint before doing anything
            self._load_checkpoint()

            # perform validation before training
            # currently, we only support validation using the reward_function.
            if self.val_reward_fn is not None and self.config.trainer.val_before_train:
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)
                if self.config.trainer.val_only:
                    return

            if self.config.algorithm.online_filtering:
                self.data_iterator = iter(self.train_dataloader)
                while self.global_step < self.training_steps:
                    self.global_step += 1

                    metrics, timing_raw = {}, {}
                    with timer("step", timing_raw):
                        with timer("gen", timing_raw):
                            batch = self._make_batch_data_online_filtering(metrics)

                        with timer("reward", timing_raw):
                            reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(batch))

                        self._maybe_update_replay_buffer(batch, reward_metrics)
                        self._maybe_trace_step(batch, reward_tensor, reward_metrics)
                        _set_batch_tensor(batch, "token_level_scores", reward_tensor)

                        metrics.update(self._structured_metrics_for_batch(batch, reward_metrics, prefix="train"))
                        metrics.update(self._experiment_metrics)

                        self._balance_batch(batch, metrics=metrics)
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with timer("old", timing_raw):
                            old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                            batch = batch.union(old_log_probs)

                        if self.use_reference_policy:
                            with timer("ref", timing_raw):
                                ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                                batch = batch.union(ref_log_probs)

                        if self.use_critic:
                            with timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with timer("adv", timing_raw):
                            metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()})

                            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                                batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                _set_batch_tensor(batch, "token_level_rewards", batch.batch["token_level_scores"])

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                            )

                        if self.use_critic:
                            with timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)

                            metrics.update(reduce_metrics(critic_output.non_tensor_batch))

                        if self.config.trainer.critic_warmup <= self.global_step:
                            with timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)

                            metrics.update(reduce_metrics(actor_output.non_tensor_batch))
                            self._maybe_run_replay_update(metrics)

                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.val_freq > 0
                            and self.global_step % self.config.trainer.val_freq == 0
                        ):
                            with timer("validation", timing_raw):
                                val_metrics = self._validate()

                            metrics.update(val_metrics)

                        if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                            with timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                    num_gpus = self.resource_pool_manager.get_num_gpus()
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
                    self.logger.log(data=metrics, step=self.global_step)
            else:
                for _ in tqdm(range(self.config.trainer.total_epochs), desc="Epoch", position=0):
                    for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):
                        self.global_step += 1
                        if self.global_step > self.training_steps:
                            break

                        metrics, timing_raw = {}, {}
                        batch: DataProto = DataProto.from_single_dict(batch_dict)

                        if "multi_modal_data" in batch.non_tensor_batch.keys():
                            mm = batch.non_tensor_batch.get("multi_modal_data", None)
                            if mm is not None and "image_1_pil" not in batch.non_tensor_batch:
                                image_1_pil = []
                                for item in mm:
                                    img0 = None
                                    if isinstance(item, dict):
                                        imgs = item.get("image", None)
                                        if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                                            img0 = imgs[0]
                                    image_1_pil.append(img0)
                                batch.non_tensor_batch["image_1_pil"] = np.array(image_1_pil, dtype=object)
                            gen_batch = batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                            )
                        else:
                            gen_batch = batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids"],
                            )

                        with timer("step", timing_raw):
                            with timer("gen", timing_raw):
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                            if self.config.algorithm.adv_estimator == "remax":
                                with timer("gen_max", timing_raw):
                                    gen_baseline_batch = deepcopy(gen_batch)
                                    gen_baseline_batch.meta_info["temperature"] = 0
                                    gen_baseline_batch.meta_info["n"] = 1
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                    batch = batch.union(gen_baseline_output)

                                    reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
                                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                                    _set_batch_tensor(batch, "reward_baselines", reward_baseline_tensor)
                                    del gen_baseline_batch, gen_baseline_output

                            batch.non_tensor_batch["uid"] = np.array(
                                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                            )
                            batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)

                            batch = batch.union(gen_batch_output)
                            if "metadata" not in batch.non_tensor_batch:
                                raise KeyError("Structured ChartQA training expects metadata in the RL batch.")
                            batch, _ = self._process_structured_chartqa_batch(batch)

                            batch.non_tensor_batch.pop("multi_modal_data", None)

                            with timer("reward", timing_raw):
                                reward_ref = self.reward_fn.compute_reward.remote(batch)

                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            self._maybe_update_replay_buffer(batch, reward_metrics)
                            self._maybe_trace_step(batch, reward_tensor, reward_metrics)
                            _set_batch_tensor(batch, "token_level_scores", reward_tensor)

                        metrics.update(self._structured_metrics_for_batch(batch, reward_metrics, prefix="train"))
                        metrics.update(self._experiment_metrics)

                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        self._balance_batch(batch, metrics=metrics)
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with timer("old", timing_raw):
                            old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                            batch = batch.union(old_log_probs)

                        if self.use_reference_policy:
                            with timer("ref", timing_raw):
                                ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                                batch = batch.union(ref_log_probs)

                        if self.use_critic:
                            with timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with timer("adv", timing_raw):
                            metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()})

                            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                                batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                _set_batch_tensor(batch, "token_level_rewards", batch.batch["token_level_scores"])

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                            )

                        if self.use_critic:
                            with timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)

                            metrics.update(reduce_metrics(critic_output.non_tensor_batch))

                        if self.config.trainer.critic_warmup <= self.global_step:
                            with timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)

                            metrics.update(reduce_metrics(actor_output.non_tensor_batch))
                            self._maybe_run_replay_update(metrics)

                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.val_freq > 0
                            and self.global_step % self.config.trainer.val_freq == 0
                        ):
                            with timer("validation", timing_raw):
                                val_metrics = self._validate()

                            metrics.update(val_metrics)

                        if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                            with timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                        num_gpus = self.resource_pool_manager.get_num_gpus()
                        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
                        self.logger.log(data=metrics, step=self.global_step)

            # perform validation after training
            if self.val_reward_fn is not None:
                if (
                    val_metrics is None
                    or self.config.trainer.val_freq <= 0
                    or self.global_step % self.config.trainer.val_freq != 0
                ):
                    val_metrics = self._validate()
                    self.logger.log(data=val_metrics, step=self.global_step)

                print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

            if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
                self._save_checkpoint()
        finally:
            if self.logger is not None:
                self.logger.finish()
