from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Tuple

import torch


RL_LORA_CHECKPOINT_VERSION = "rl_lora_v1"
_PEFT_BASE_PREFIX = "base_model.model."
_LORA_PART_RE = re.compile(r"^(?P<prefix>.+)\.lora_(?P<part>A|B)\.(?P<adapter>[^.]+)\.weight$")
_ADAPTER_KEY_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
    ".lora_magnitude_vector.",
)


def _to_plain_config_dict(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        data = asdict(config)
    elif hasattr(config, "__dict__"):
        data = dict(vars(config))
    else:
        raise TypeError(f"Unsupported RL LoRA config type: {type(config)!r}")

    target_modules = data.get("target_modules", ())
    data["target_modules"] = list(target_modules)
    return data


def _to_full_tensor(value: Any) -> torch.Tensor:
    tensor = value.full_tensor() if hasattr(value, "full_tensor") else value
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor-like value, got {type(tensor)!r}")
    return tensor


def _require_peft():
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError(
            "RL LoRA training requires `peft`. Install project dependencies before enabling "
            "`worker.actor.rl_lora.enable=true`."
        ) from exc
    return LoraConfig, get_peft_model


def is_rl_lora_model(model: Any) -> bool:
    return hasattr(model, "peft_config") and hasattr(model, "active_adapter")


def is_adapter_state_key(name: str) -> bool:
    return any(marker in name for marker in _ADAPTER_KEY_MARKERS)


def normalize_peft_weight_name(name: str) -> str:
    if name.startswith(_PEFT_BASE_PREFIX):
        name = name[len(_PEFT_BASE_PREFIX) :]
    return name.replace(".base_layer.", ".")


def get_trainable_parameter_names(model: torch.nn.Module) -> Tuple[str, ...]:
    return tuple(name for name, param in model.named_parameters() if param.requires_grad)


def prepare_rl_lora_model(model: torch.nn.Module, rl_lora_config: Any, base_model_path: str) -> torch.nn.Module:
    config_dict = _to_plain_config_dict(rl_lora_config)
    if not config_dict.get("enable", False):
        return model

    LoraConfig, get_peft_model = _require_peft()
    peft_config = LoraConfig(
        r=int(config_dict["r"]),
        lora_alpha=int(config_dict["alpha"]),
        lora_dropout=float(config_dict["dropout"]),
        target_modules=list(config_dict["target_modules"]),
        bias="none",
    )
    peft_config.base_model_name_or_path = base_model_path
    model = get_peft_model(model, peft_config)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    return model


def create_adapter_state_dict(model_state_dict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    adapter_state: Dict[str, torch.Tensor] = {}
    for name, value in model_state_dict.items():
        if not is_adapter_state_key(name):
            continue
        adapter_state[name] = _to_full_tensor(value).detach().cpu()
    if not adapter_state:
        raise ValueError("No RL LoRA adapter weights were found in the actor state dict.")
    return adapter_state


def _scaling_for_config(rl_lora_config: Any) -> float:
    config_dict = _to_plain_config_dict(rl_lora_config)
    rank = int(config_dict["r"])
    if rank <= 0:
        raise ValueError(f"RL LoRA rank must be positive, got {rank}.")
    return float(config_dict["alpha"]) / float(rank)


def iter_merged_weight_items(
    model_state_dict: Mapping[str, Any],
    rl_lora_config: Any,
) -> Iterator[Tuple[str, torch.Tensor]]:
    scaling = _scaling_for_config(rl_lora_config)
    lora_parts: MutableMapping[str, Dict[str, torch.Tensor]] = {}

    for name, value in model_state_dict.items():
        match = _LORA_PART_RE.match(name)
        if match is None:
            continue
        target_name = normalize_peft_weight_name(f"{match.group('prefix')}.weight")
        lora_parts.setdefault(target_name, {})[match.group("part")] = _to_full_tensor(value)

    yielded_targets = set()
    for name, value in model_state_dict.items():
        if is_adapter_state_key(name):
            continue

        normalized_name = normalize_peft_weight_name(name)
        tensor = _to_full_tensor(value)
        if normalized_name in lora_parts:
            parts = lora_parts[normalized_name]
            if "A" not in parts or "B" not in parts:
                raise ValueError(f"Incomplete RL LoRA factors found for {normalized_name}.")
            delta = torch.matmul(parts["B"].float(), parts["A"].float()).to(dtype=tensor.dtype, device=tensor.device)
            tensor = tensor + delta * scaling
            yielded_targets.add(normalized_name)

        yield normalized_name, tensor

    missing_targets = set(lora_parts) - yielded_targets
    if missing_targets:
        missing = ", ".join(sorted(missing_targets))
        raise KeyError(f"Missing base weights for RL LoRA targets: {missing}")


def build_rl_lora_checkpoint_metadata(base_model_path: str, rl_lora_config: Any) -> Dict[str, Any]:
    return {
        "format_version": RL_LORA_CHECKPOINT_VERSION,
        "base_model_path": os.path.abspath(base_model_path),
        "rl_lora": _to_plain_config_dict(rl_lora_config),
    }


def ensure_rl_lora_checkpoint_dir(path: os.PathLike[str] | str) -> Dict[str, Any]:
    checkpoint_path = Path(path)
    metadata_path = checkpoint_path / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(
            "Unsupported checkpoint format. RL-only LoRA resume expects an adapter-first checkpoint "
            "with metadata.json under the actor directory."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != RL_LORA_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported RL LoRA checkpoint version: {metadata.get('format_version')!r}. "
            f"Expected {RL_LORA_CHECKPOINT_VERSION!r}."
        )

    adapter_dir = checkpoint_path / "adapter"
    adapter_model_path = adapter_dir / "adapter_model.bin"
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_model_path.exists() or not adapter_config_path.exists():
        raise ValueError(
            "Incomplete RL LoRA checkpoint. Expected adapter/adapter_model.bin and "
            "adapter/adapter_config.json."
        )

    return metadata


def resolve_rl_lora_adapter_dir(path: os.PathLike[str] | str) -> Path:
    candidate = Path(path)
    if (candidate / "adapter_config.json").exists():
        return candidate
    ensure_rl_lora_checkpoint_dir(candidate)
    return candidate / "adapter"
