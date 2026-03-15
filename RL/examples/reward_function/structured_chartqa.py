from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp


REWARD_DIR = Path(__file__).resolve().parent
RL_ROOT = REWARD_DIR.parents[1]
PROJECT_ROOT = REWARD_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.chartqa.common import compute_chartqa_match_score

JUDGE_DIR = RL_ROOT / "judge"
JUDGE_PROMPT_PATH = JUDGE_DIR / "judge_prompt.txt"
JUDGE_INFO_PATHS = [
    JUDGE_DIR / "judge_info.json",
    JUDGE_DIR / "judge_info.example.json",
]

_CACHE_BY_PATH: dict[str, "JudgeCache"] = {}


def extract_final_answer(text: str) -> str:
    content = (text or "").strip()
    if not content:
        return ""
    matches = re.findall(r"FINAL ANSWER:\s*(.*?)(?=\n|$)", content, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().strip(".")
    return content.splitlines()[-1].strip().strip(".")


def compute_rule_score(answer_text: str, ground_truth: str) -> float:
    answer = extract_final_answer(answer_text)
    if not answer:
        return 0.0
    return compute_chartqa_match_score(answer, ground_truth)


class JudgeCache:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, float] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self.entries[str(record["key"])] = float(record["score"])

    def get(self, key: str) -> Optional[float]:
        return self.entries.get(key)

    def set_many(self, updates: Dict[str, float]) -> None:
        if not updates:
            return
        with self.path.open("a", encoding="utf-8") as file:
            for key, score in updates.items():
                if key in self.entries:
                    continue
                self.entries[key] = float(score)
                file.write(json.dumps({"key": key, "score": float(score)}, ensure_ascii=False) + "\n")


def get_judge_cache(path: str | Path) -> JudgeCache:
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in _CACHE_BY_PATH:
        _CACHE_BY_PATH[resolved] = JudgeCache(resolved)
    return _CACHE_BY_PATH[resolved]


def judge_cache_key(query: str, ground_truth: str, answer: str) -> str:
    payload = json.dumps(
        {
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


async def _fetch(session, prompt, semaphore, url, headers=None, openrouter_model=None):
    async with semaphore:
        if headers is not None and openrouter_model is not None:
            payload = {
                "model": openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 64,
                "stream": False,
            }
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    raw = await response.json()
                    choices = raw.get("choices") if isinstance(raw, dict) else None
                    if isinstance(choices, list) and choices:
                        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                        return str(message.get("content") or "")
                    return ""
                return await response.text()

        payload = {
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 64,
        }
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return str(result.get("text", [""])[0])
            return await response.text()


async def _judge_prompts(prompts: List[str], url: str, headers=None, openrouter_model=None) -> List[str]:
    semaphore = asyncio.Semaphore(64)
    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch(session, prompt, semaphore, url, headers=headers, openrouter_model=openrouter_model)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)


def _resolve_judge_runtime():
    judge_info_path = next((path for path in JUDGE_INFO_PATHS if path.exists()), None)
    if judge_info_path is None:
        raise FileNotFoundError("judge_info.json not found. Copy judge_info.example.json to judge_info.json first.")
    with judge_info_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    host = data.get("host")
    openrouter_api_key = (data.get("openrouter_api_key") or data.get("openrouter_key") or "").strip()
    openrouter_model = (data.get("openrouter_model") or "deepseek/deepseek-v3.2").strip()
    openrouter_base_url = (data.get("openrouter_base_url") or "https://openrouter.ai/api/v1/chat/completions").strip()
    openrouter_http_referer = (data.get("openrouter_http_referer") or "").strip()
    openrouter_app_title = (data.get("openrouter_app_title") or "chartqa_rl_judge").strip()

    headers = None
    if openrouter_api_key:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openrouter_api_key}"}
        if openrouter_http_referer:
            headers["HTTP-Referer"] = openrouter_http_referer
        if openrouter_app_title:
            headers["X-Title"] = openrouter_app_title
        return openrouter_base_url, headers, openrouter_model

    return f"http://{host}:7999/generate", None, None


def _extract_judge_yes(text: str) -> float:
    matches = re.findall(r"<(.*?)>", text)
    if matches:
        return 1.0 if matches[-1] == "|YES|" else 0.0
    return 0.0


def batch_judge_scores(records: List[Dict[str, str]], cache_path: str | Path) -> List[float]:
    if not records:
        return []
    cache = get_judge_cache(cache_path)
    with JUDGE_PROMPT_PATH.open("r", encoding="utf-8") as file:
        judge_prompt = file.read()

    results: List[float] = [0.0] * len(records)
    missing_indices: List[int] = []
    prompts: List[str] = []
    keys: List[str] = []

    for index, record in enumerate(records):
        answer = extract_final_answer(record["answer"])
        key = judge_cache_key(record["query"], record["ground_truth"], answer)
        cached = cache.get(key)
        if cached is not None:
            results[index] = cached
            continue
        missing_indices.append(index)
        keys.append(key)
        prompts.append(
            judge_prompt.replace("<question>", record["query"])
            .replace("<gt>", str(record["ground_truth"]))
            .replace("<answer>", answer)
        )

    if prompts:
        url, headers, openrouter_model = _resolve_judge_runtime()
        responses = asyncio.run(_judge_prompts(prompts, url, headers=headers, openrouter_model=openrouter_model))
        updates: Dict[str, float] = {}
        for local_index, response_text in enumerate(responses):
            score = _extract_judge_yes(response_text)
            index = missing_indices[local_index]
            results[index] = score
            updates[keys[local_index]] = score
        cache.set_many(updates)

    return results


def compute_structured_scores(
    records: List[Dict[str, object]],
    rule_weight: float = 0.4,
    judge_weight: float = 0.6,
    tool_gain_weight: float = 0.75,
    invalid_penalty: float = 1.0,
    ineffective_penalty: float = 0.25,
    judge_cache_path: str | Path = "./judge/cache/structured_chartqa_judge_cache.jsonl",
) -> List[Dict[str, float]]:
    final_judge_records = []
    baseline_judge_records = []
    for record in records:
        final_judge_records.append(
            {
                "query": str(record["query"]),
                "ground_truth": str(record["ground_truth"]),
                "answer": str(record.get("final_answer_text") or ""),
            }
        )
        baseline_judge_records.append(
            {
                "query": str(record["query"]),
                "ground_truth": str(record["ground_truth"]),
                "answer": str(record.get("baseline_answer_text") or ""),
            }
        )

    if judge_weight > 0:
        judge_scores = batch_judge_scores(final_judge_records, cache_path=judge_cache_path)
        baseline_judge_scores = batch_judge_scores(baseline_judge_records, cache_path=judge_cache_path)
    else:
        judge_scores = [0.0] * len(records)
        baseline_judge_scores = [0.0] * len(records)

    outputs: List[Dict[str, float]] = []
    for index, record in enumerate(records):
        final_text = str(record.get("final_answer_text") or "")
        baseline_text = str(record.get("baseline_answer_text") or "")
        ground_truth = str(record["ground_truth"])

        rule_score = compute_rule_score(final_text, ground_truth)
        baseline_rule_score = compute_rule_score(baseline_text, ground_truth)
        judge_score = judge_scores[index]
        baseline_judge_score = baseline_judge_scores[index]
        final_mix = rule_weight * rule_score + judge_weight * judge_score
        baseline_mix = rule_weight * baseline_rule_score + judge_weight * baseline_judge_score

        tool_requested = float(bool(record.get("tool_requested", False)))
        tool_executed = float(bool(record.get("tool_executed", False)))
        invalid_action_flag = float(bool(record.get("invalid_action", False)))
        tool_cost = float(record.get("tool_cost", 0.0) or 0.0)

        if tool_requested and tool_executed:
            tool_gain = final_mix - baseline_mix
        else:
            tool_gain = 0.0

        invalid_term = invalid_penalty if invalid_action_flag else 0.0
        ineffective_flag = float(tool_requested and tool_executed and tool_gain <= 0.0)
        ineffective_term = ineffective_penalty if ineffective_flag else 0.0
        overall = final_mix + tool_gain_weight * tool_gain - tool_cost - invalid_term - ineffective_term
        effective_tool = float(tool_requested and tool_executed and tool_gain > 0.05)

        outputs.append(
            {
                "overall": overall,
                "rule_score": rule_score,
                "judge_score": judge_score,
                "baseline_rule_score": baseline_rule_score,
                "baseline_judge_score": baseline_judge_score,
                "tool_gain": tool_gain,
                "tool_cost": tool_cost,
                "invalid_action": invalid_action_flag,
                "tool_executed": tool_executed,
                "effective_tool": effective_tool,
                "answer_accuracy": rule_score,
                "final_mix": final_mix,
                "baseline_mix": baseline_mix,
            }
        )
    return outputs
