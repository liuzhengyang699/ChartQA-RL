import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizer, ProcessorMixin

from ...rl_lora import (
    build_rl_lora_checkpoint_metadata,
    create_adapter_state_dict,
    ensure_rl_lora_checkpoint_dir,
    is_adapter_state_key,
)
from .checkpoint_manager import BaseCheckpointManager


class FSDPRLLoRACheckpointManager(BaseCheckpointManager):
    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
        base_model_path: str,
        rl_lora_config: Any,
    ):
        super().__init__(model, optimizer, lr_scheduler, processing_class)
        self.base_model_path = os.path.abspath(base_model_path)
        self.rl_lora_config = rl_lora_config

    def _optimizer_path(self, path: str) -> str:
        return os.path.join(path, f"optimizer_rank_{self.rank}.pt")

    def _extra_path(self, path: str) -> str:
        return os.path.join(path, f"extra_state_rank_{self.rank}.pt")

    def _adapter_dir(self, path: str) -> str:
        return os.path.join(path, "adapter")

    def _metadata_path(self, path: str) -> str:
        return os.path.join(path, "metadata.json")

    def _current_rl_lora_config_dict(self) -> Dict[str, Any]:
        if is_dataclass(self.rl_lora_config):
            data = asdict(self.rl_lora_config)
        else:
            data = dict(vars(self.rl_lora_config))
        data["target_modules"] = list(data.get("target_modules", ()))
        return data

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            return

        metadata = ensure_rl_lora_checkpoint_dir(path)
        if os.path.abspath(metadata["base_model_path"]) != self.base_model_path:
            raise ValueError(
                "RL LoRA checkpoint base model does not match the current actor model path. "
                f"checkpoint={metadata['base_model_path']}, current={self.base_model_path}"
            )

        checkpoint_config = metadata.get("rl_lora", {})
        current_config = self._current_rl_lora_config_dict()
        if checkpoint_config != current_config:
            raise ValueError(
                "RL LoRA checkpoint config does not match the current actor RL LoRA config. "
                f"checkpoint={checkpoint_config}, current={current_config}"
            )

        adapter_path = os.path.join(self._adapter_dir(path), "adapter_model.bin")
        optim_path = self._optimizer_path(path)
        extra_path = self._extra_path(path)
        print(f"[rank-{self.rank}]: Loading RL LoRA adapter from {os.path.abspath(adapter_path)}.")
        print(f"[rank-{self.rank}]: Loading optimizer from {os.path.abspath(optim_path)}.")
        print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")

        adapter_state_dict = torch.load(adapter_path, map_location="cpu", weights_only=False)
        optim_state_dict = torch.load(optim_path, map_location="cpu", weights_only=False)
        extra_state_dict = torch.load(extra_path, map_location="cpu", weights_only=False)

        with FSDP.summon_full_params(self.model, recurse=True, writeback=True, offload_to_cpu=True):
            wrapped_model = self.model._fsdp_wrapped_module
            expected_adapter_keys = {name for name, _ in wrapped_model.named_parameters() if is_adapter_state_key(name)}
            missing_adapter_keys = expected_adapter_keys - set(adapter_state_dict)
            unexpected_adapter_keys = set(adapter_state_dict) - expected_adapter_keys
            if missing_adapter_keys or unexpected_adapter_keys:
                raise ValueError(
                    "RL LoRA checkpoint adapter weights do not match the current actor structure. "
                    f"missing={sorted(missing_adapter_keys)}, unexpected={sorted(unexpected_adapter_keys)}"
                )
            wrapped_model.load_state_dict(adapter_state_dict, strict=False)

        self.optimizer.load_state_dict(optim_state_dict)
        self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

    def save_checkpoint(self, path: str):
        path = self.local_mkdir(path)
        dist.barrier()

        optimizer_path = self._optimizer_path(path)
        extra_path = self._extra_path(path)
        optimizer_state = self.optimizer.state_dict()
        extra_state_dict = {
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "rng": self.get_rng_state(),
        }

        print(f"[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optimizer_path)}.")
        print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}.")
        torch.save(optimizer_state, optimizer_path)
        torch.save(extra_state_dict, extra_path)

        model_state_dict = get_model_state_dict(self.model)
        adapter_state_dict = create_adapter_state_dict(model_state_dict)
        del model_state_dict

        if self.rank == 0:
            adapter_dir = self.local_mkdir(self._adapter_dir(path))
            metadata = build_rl_lora_checkpoint_metadata(self.base_model_path, self.rl_lora_config)
            wrapped_model = self.model._fsdp_wrapped_module
            active_adapter = getattr(wrapped_model, "active_adapter", "default")
            if isinstance(active_adapter, list):
                active_adapter = active_adapter[0]

            wrapped_model.peft_config[active_adapter].save_pretrained(adapter_dir)
            torch.save(adapter_state_dict, os.path.join(adapter_dir, "adapter_model.bin"))
            with open(self._metadata_path(path), "w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)

        dist.barrier()
