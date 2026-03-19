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

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def _safe_mean(values: Sequence[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def compute_structured_metrics(
    reward_metrics: Dict[str, Sequence[float]],
    action_valid: Sequence[bool],
    tool_requested: Sequence[bool],
    tool_exec_success: Sequence[bool],
    invalid_action: Sequence[bool],
    prefix: str,
) -> Dict[str, float]:
    action_valid_arr = np.asarray(action_valid, dtype=bool)
    tool_requested_arr = np.asarray(tool_requested, dtype=bool)
    tool_exec_success_arr = np.asarray(tool_exec_success, dtype=bool)
    invalid_action_arr = np.asarray(invalid_action, dtype=bool)

    total = float(max(len(action_valid_arr), 1))
    tool_legal = float(np.sum(action_valid_arr & tool_requested_arr))
    tool_exec_success_count = float(np.sum(tool_exec_success_arr))
    executed_indices = np.flatnonzero(tool_exec_success_arr)

    tool_gain_values = reward_metrics.get("tool_gain", [])
    effective_tool_values = reward_metrics.get("effective_tool", [])
    avg_tool_gain = (
        float(np.mean([tool_gain_values[index] for index in executed_indices])) if len(executed_indices) > 0 else 0.0
    )
    tool_effectiveness = (
        float(np.mean([effective_tool_values[index] for index in executed_indices]))
        if len(executed_indices) > 0 and len(effective_tool_values) > 0
        else 0.0
    )

    metrics = {
        f"{prefix}/QAAccuracy": _safe_mean(reward_metrics.get("answer_accuracy", [])),
        f"{prefix}/ToolCallRate": float(np.sum(tool_requested_arr)) / total,
        f"{prefix}/LegalActionRate": float(np.sum(action_valid_arr)) / total,
        f"{prefix}/ToolExecSuccessRate": tool_exec_success_count / max(tool_legal, 1.0),
        f"{prefix}/ToolEffectivenessRate": tool_effectiveness,
        f"{prefix}/AvgToolGain": avg_tool_gain,
        f"{prefix}/InvalidActionRate": float(np.sum(invalid_action_arr)) / total,
        f"{prefix}/FinalMix": _safe_mean(reward_metrics.get("final_mix", [])),
        f"{prefix}/BaselineMix": _safe_mean(reward_metrics.get("baseline_mix", [])),
        f"{prefix}/RuleScore": _safe_mean(reward_metrics.get("rule_score", [])),
        f"{prefix}/JudgeScore": _safe_mean(reward_metrics.get("judge_score", [])),
        f"{prefix}/BaselineRuleScore": _safe_mean(reward_metrics.get("baseline_rule_score", [])),
        f"{prefix}/BaselineJudgeScore": _safe_mean(reward_metrics.get("baseline_judge_score", [])),
        f"{prefix}/RewardScore": _safe_mean(reward_metrics.get("overall", [])),
    }
    return metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> Dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    response_mask = batch.batch["attention_mask"][:, -batch.batch["responses"].size(-1) :].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
    }
    return metrics
