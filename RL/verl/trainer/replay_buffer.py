from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


class ReplayBuffer:
    def __init__(
        self,
        buffer_dir: str | Path,
        buffer_size: int = 50000,
        per_figure_limit: int = 3,
        min_final_mix: float = 0.8,
        min_tool_gain: float = 0.1,
        seed: int = 42,
    ):
        self.buffer_dir = Path(buffer_dir).expanduser().resolve()
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_path = self.buffer_dir / "buffer.jsonl"
        self.buffer_size = int(buffer_size)
        self.per_figure_limit = int(per_figure_limit)
        self.min_final_mix = float(min_final_mix)
        self.min_tool_gain = float(min_tool_gain)
        self.random = random.Random(seed)
        self.entries: Dict[str, Dict[str, object]] = {}
        self._load()

    def _load(self) -> None:
        if not self.buffer_path.exists():
            return
        with self.buffer_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = self._entry_key(entry)
                self.entries[key] = entry
        self._prune()

    def _entry_key(self, entry: Dict[str, object]) -> str:
        return json.dumps(
            [
                entry.get("figure_id"),
                entry.get("action_target_json"),
                entry.get("answer_target_text"),
            ],
            ensure_ascii=False,
            sort_keys=True,
        )

    def _passes_threshold(self, entry: Dict[str, object]) -> bool:
        decision = str(entry.get("tool_metadata", {}).get("decision", "direct"))
        final_mix = float(entry.get("final_mix", 0.0) or 0.0)
        tool_gain = float(entry.get("tool_gain", 0.0) or 0.0)
        if decision == "direct":
            return final_mix >= 0.9
        return final_mix >= self.min_final_mix and tool_gain >= self.min_tool_gain

    def _score(self, entry: Dict[str, object]) -> float:
        return float(entry.get("quality_score", 0.0) or 0.0)

    def _prune(self) -> None:
        per_figure: Dict[str, List[Dict[str, object]]] = {}
        for entry in self.entries.values():
            figure_id = str(entry.get("figure_id"))
            per_figure.setdefault(figure_id, []).append(entry)

        kept: Dict[str, Dict[str, object]] = {}
        for figure_entries in per_figure.values():
            for entry in sorted(figure_entries, key=self._score, reverse=True)[: self.per_figure_limit]:
                kept[self._entry_key(entry)] = entry

        if len(kept) > self.buffer_size:
            top_entries = sorted(kept.values(), key=self._score, reverse=True)[: self.buffer_size]
            kept = {self._entry_key(entry): entry for entry in top_entries}

        self.entries = kept

    def _persist(self) -> None:
        with self.buffer_path.open("w", encoding="utf-8") as file:
            for entry in sorted(self.entries.values(), key=self._score, reverse=True):
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def add_entries(self, entries: Sequence[Dict[str, object]]) -> int:
        added = 0
        for entry in entries:
            if not self._passes_threshold(entry):
                continue
            key = self._entry_key(entry)
            existing = self.entries.get(key)
            if existing is None or self._score(entry) > self._score(existing):
                self.entries[key] = dict(entry)
                added += 1
        if added:
            self._prune()
            self._persist()
        return added

    def __len__(self) -> int:
        return len(self.entries)

    def _bucketed_entries(self, require_answer: bool = False) -> Dict[str, List[Dict[str, object]]]:
        buckets: Dict[str, List[Dict[str, object]]] = {
            "tool_positive": [],
            "direct_high_confidence": [],
            "hard_negative_repaired": [],
        }
        for entry in self.entries.values():
            if require_answer and not entry.get("answer_target_text"):
                continue
            bucket = str(entry.get("bucket", "tool_positive"))
            buckets.setdefault(bucket, []).append(entry)
        return buckets

    def _sample_with_ratios(self, count: int, require_answer: bool = False) -> List[Dict[str, object]]:
        if count <= 0 or len(self.entries) == 0:
            return []
        buckets = self._bucketed_entries(require_answer=require_answer)
        desired = {
            "tool_positive": int(round(count * 0.5)),
            "direct_high_confidence": int(round(count * 0.3)),
        }
        desired["hard_negative_repaired"] = max(0, count - desired["tool_positive"] - desired["direct_high_confidence"])

        sampled: List[Dict[str, object]] = []
        for bucket_name, bucket_count in desired.items():
            pool = buckets.get(bucket_name, [])
            if not pool:
                continue
            if len(pool) <= bucket_count:
                sampled.extend(pool)
            else:
                sampled.extend(self.random.sample(pool, bucket_count))

        if len(sampled) < count:
            remaining_pool = [entry for entry in self.entries.values() if (not require_answer or entry.get("answer_target_text"))]
            remaining_pool = [entry for entry in remaining_pool if entry not in sampled]
            needed = min(count - len(sampled), len(remaining_pool))
            if needed > 0:
                sampled.extend(self.random.sample(remaining_pool, needed))
        return sampled

    def sample_supervision(
        self,
        total_batch_size: int,
        action_weight: float,
        answer_weight: float,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        if total_batch_size <= 0 or len(self.entries) == 0:
            return [], []
        total_weight = max(action_weight + answer_weight, 1e-8)
        action_count = int(round(total_batch_size * (action_weight / total_weight)))
        action_count = min(max(action_count, 1), total_batch_size)
        answer_count = max(total_batch_size - action_count, 1)
        action_entries = self._sample_with_ratios(action_count, require_answer=False)
        answer_entries = self._sample_with_ratios(answer_count, require_answer=True)
        return action_entries, answer_entries
