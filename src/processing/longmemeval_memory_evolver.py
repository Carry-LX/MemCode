import json
import os
import asyncio
import uuid
import threading
import sqlite3
import numpy as np
import math
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter
from collections import defaultdict
from datetime import datetime

# 引入项目依赖
from src.core.memory_structures import AtomicMemoryUnit
from src.core.metadata_db_handler import MetadataDBHandler
from src.core.vector_db_handler import VectorDBHandler
from src.utils.llm_interface import OpenAIInterface

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    raise ImportError("错误: 记忆演化依赖 'scikit-learn'，请安装: pip install scikit-learn")


class MemoryConsolidator:
    """Asynchronous LongMemEval consolidation with speaker/status-aware grouping.

    Conservative cluster policies preserve count, update and preference
    evidence. Archive decisions retain source pointers and superseded_by
    redirects. Time spans are stored in metadata, not injected into fact text."""

    # 超参数配置
    SIMILARITY_THRESHOLD = 0.80
    MIN_CLUSTER_SIZE = 2
    MAX_CLUSTER_SIZE = 50
    SAFE_ARCHIVE_MAX_SPAN_HOURS = 72.0
    SAFE_ARCHIVE_MAX_PARENTS = 3
    SAFE_ARCHIVE_MAX_NEW_FACT_LEN = 220
    AGGREGATE_MAX_CLUSTER_SIZE = 8
    AGGREGATE_MAX_SPAN_HOURS = 24 * 21

    def __init__(self,
                 meta_db: MetadataDBHandler,
                 vec_db: VectorDBHandler,
                 embedder: Any,
                 fact_num: int,
                 llm: OpenAIInterface,
                 vq_model: Any = None,
                 max_concurrency: int = 8):

        self.meta_db = meta_db
        self.db_path = meta_db.db_path
        self.vec_db = vec_db
        self.embedder = embedder
        self.llm = llm
        self.vq_model = vq_model

        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.write_lock = threading.Lock()
        self._base_speaker_cache: Dict[str, str] = {}

        print("  > [Config] LongMemEval consolidation: HAC, source pointers and metadata time spans")

    # =========================================================================
    # [Helper] 基础工具
    # =========================================================================
    def _calculate_idf(self, code: int, stats: Dict[str, Any]) -> float:
        total = stats.get("total_docs", 0)
        if total <= 0: return 0.0
        freq = stats.get("code_freq", {}).get(code, 0)
        return math.log(total / (freq + 1))

    def _get_detailed_db_stats(self) -> Dict[str, Any]:
        stats = {"total_active": 0, "total_archived": 0}
        if not os.path.exists(self.db_path): return stats
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT metadata FROM atomic_memory")
            for (m_str,) in cur.fetchall():
                if not m_str: stats["total_active"] += 1; continue
                try:
                    m = json.loads(m_str)
                    if m.get('status') == 'archived':
                        stats["total_archived"] += 1
                    else:
                        stats["total_active"] += 1
                except:
                    stats["total_active"] += 1
            conn.close()
        except:
            pass
        return stats

    def _normalize_speaker(self, speaker: Any) -> str:
        value = str(speaker or "").strip().lower()
        if not value:
            return "neutral"
        if "assistant" in value or value == "ai":
            return "assistant"
        if "user" in value or "human" in value:
            return "user"
        return value

    def _get_base_speaker(self, base_id: str) -> str:
        if not base_id:
            return "neutral"
        if base_id in self._base_speaker_cache:
            return self._base_speaker_cache[base_id]

        speaker = "neutral"
        try:
            base_mem = self.meta_db.get_base_memory_by_id(base_id)
            if base_mem and getattr(base_mem, "metadata", None):
                speaker = self._normalize_speaker(base_mem.metadata.get("speaker"))
        except Exception:
            speaker = "neutral"

        self._base_speaker_cache[base_id] = speaker
        return speaker

    def _infer_speaker_bucket(self, fact: AtomicMemoryUnit) -> str:
        if fact.metadata:
            speaker = self._normalize_speaker(fact.metadata.get("speaker"))
            if speaker != "neutral":
                return speaker

        text = (fact.fact_text or "").strip().lower()
        if text.startswith("user"):
            return "user"
        if text.startswith("assistant"):
            return "assistant"

        base_speakers = {
            self._get_base_speaker(base_id)
            for base_id in (fact.source_base_ids or [])
            if base_id
        }
        base_speakers.discard("neutral")
        if len(base_speakers) == 1:
            return next(iter(base_speakers))
        return "neutral"

    def _parse_timestamp(self, raw_ts: Any) -> Optional[datetime]:
        if not raw_ts:
            return None
        text = str(raw_ts).strip()
        if not text or text == "Unknown":
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(text[:19] if fmt.endswith("%S") else text[:16], fmt)
            except ValueError:
                continue
        return None

    def _estimate_time_span_hours(self, facts: List[AtomicMemoryUnit]) -> float:
        timestamps = [
            self._parse_timestamp((fact.metadata or {}).get("timestamp"))
            for fact in facts
        ]
        timestamps = [ts for ts in timestamps if ts is not None]
        if len(timestamps) < 2:
            return 0.0
        return (max(timestamps) - min(timestamps)).total_seconds() / 3600.0

    def _split_facts_by_speaker(self, facts: List[AtomicMemoryUnit]) -> Dict[str, List[AtomicMemoryUnit]]:
        buckets: Dict[str, List[AtomicMemoryUnit]] = defaultdict(list)
        for fact in facts:
            buckets[self._infer_speaker_bucket(fact)].append(fact)
        return buckets

    def _infer_fact_status(self, fact: AtomicMemoryUnit) -> str:
        text = (fact.fact_text or "").strip().lower()
        speaker = self._infer_speaker_bucket(fact)

        advice_markers = [
            "assistant advised", "assistant suggested", "assistant recommended", "assistant proposed",
            "assistant encouraged", "assistant explained", "assistant shared", "assistant outlined",
        ]
        planned_markers = [
            "plans to", "planning to", "is planning to", "thinking of", "considering", "wants to",
            "would like to", "hopes to", "aims to", "scheduled", "will ", "going to", "interested in trying",
        ]
        completed_markers = [
            "bought", "purchased", "redeemed", "attended", "went", "submitted", "spent", "used", "made",
            "finished", "completed", "graduated", "visited", "baked", "led", "owned", "watched", "played",
        ]
        state_markers = [
            "currently", "has been", "works as", "is responsible for", "owns", "lives", "prefers", "likes",
            "favorite", "allergic", "commute", "goal is", "long-term goal", "enjoys", "looking for",
            "trying to find", "similar to", "instead of", "rather than", "beyond", "wants recommendations",
            "not interested", "doesn't like", "does not like",
        ]

        if any(marker in text for marker in advice_markers):
            return "advice"
        if speaker == "assistant" and any(token in text for token in ["you can", "you should", "consider", "recommend", "suggest"]):
            return "advice"
        if any(marker in text for marker in planned_markers):
            return "planned"
        if any(marker in text for marker in state_markers):
            return "state"
        if any(marker in text for marker in completed_markers):
            return "completed"
        return "other"

    def _split_facts_by_speaker_and_status(self, facts: List[AtomicMemoryUnit]) -> Dict[Tuple[str, str], List[AtomicMemoryUnit]]:
        buckets: Dict[Tuple[str, str], List[AtomicMemoryUnit]] = defaultdict(list)
        for fact in facts:
            buckets[(self._infer_speaker_bucket(fact), self._infer_fact_status(fact))].append(fact)
        return buckets

    def _estimate_unique_day_count(self, facts: List[AtomicMemoryUnit]) -> int:
        days = set()
        for fact in facts:
            ts = self._parse_timestamp((fact.metadata or {}).get("timestamp"))
            if ts is not None:
                days.add(ts.date().isoformat())
        return len(days)

    def _is_count_sensitive_cluster(self, facts: List[AtomicMemoryUnit]) -> bool:
        texts = " ".join((fact.fact_text or "").lower() for fact in facts)
        count_markers = [
            "how many", "total", "spent", "$", "times", "number of", "goals", "assists",
            "projects", "items", "pieces", "appointments", "episodes", "plants", "movies",
        ]
        has_digits = any(char.isdigit() for char in texts)
        return has_digits or any(marker in texts for marker in count_markers)

    def _is_update_sensitive_cluster(self, facts: List[AtomicMemoryUnit]) -> bool:
        texts = " ".join((fact.fact_text or "").lower() for fact in facts)
        update_markers = [
            "most recent", "latest", "current", "currently", "now", "updated", "update",
            "personal best", "pb", "new record", "record time", "switched", "changed",
            "change in", "improved to", "improved from", "previously", "used to", "now uses",
        ]
        return any(marker in texts for marker in update_markers)

    def _is_preference_sensitive_cluster(self, facts: List[AtomicMemoryUnit]) -> bool:
        texts = " ".join((fact.fact_text or "").lower() for fact in facts)
        preference_markers = [
            "prefer", "preference", "likes", "dislikes", "favorite", "enjoys", "interested in",
            "would rather", "not interested", "doesn't like", "does not like", "allergic",
            "sneezing", "sensitive to", "looking for", "trying to find", "similar to", "instead of",
            "rather than", "beyond", "wants recommendations", "recommendations for",
        ]
        return any(marker in texts for marker in preference_markers)

    def _should_skip_group(self, facts: List[AtomicMemoryUnit]) -> bool:
        if len(facts) < self.MIN_CLUSTER_SIZE:
            return True

        speaker = self._infer_speaker_bucket(facts[0])
        status = self._infer_fact_status(facts[0])
        span_hours = self._estimate_time_span_hours(facts)

        # Separate speaker roles and factual status before aggregation.


        if speaker == "assistant" and status in {"advice", "planned"}:
            return True

        if status in {"planned", "advice"} and len(facts) > 3:
            return True

        # Apply count-sensitive checks to clusters, not entire speaker/status buckets.


        return False

    def _should_skip_cluster(self, facts: List[AtomicMemoryUnit]) -> bool:
        if len(facts) < self.MIN_CLUSTER_SIZE:
            return True

        speaker = self._infer_speaker_bucket(facts[0])
        status = self._infer_fact_status(facts[0])
        span_hours = self._estimate_time_span_hours(facts)
        unique_day_count = self._estimate_unique_day_count(facts)
        is_count_sensitive = self._is_count_sensitive_cluster(facts)
        is_update_sensitive = self._is_update_sensitive_cluster(facts)
        is_preference_sensitive = self._is_preference_sensitive_cluster(facts)

        if speaker == "assistant" and status in {"advice", "planned"}:
            return True

        if status in {"planned", "advice"} and len(facts) > 3:
            return True

        if is_count_sensitive and status == "completed":
            if unique_day_count > 1:
                return True
            if len(facts) > 2:
                return True
            if span_hours > 48:
                return True

        if is_count_sensitive:
            if len(facts) > self.AGGREGATE_MAX_CLUSTER_SIZE:
                return True
            if span_hours > self.AGGREGATE_MAX_SPAN_HOURS:
                return True

        if is_update_sensitive:
            if len(facts) > 2:
                return True
            if unique_day_count > 2:
                return True

        if is_preference_sensitive:
            if speaker != "user":
                return True
            if len(facts) > 2:
                return True

        return False

    def _format_fact_text_for_old_retrieval(
        self,
        text: str,
        status: str,
        mode: str = "UNKNOWN",
        is_update_sensitive: bool = False,
        is_preference_sensitive: bool = False
    ) -> str:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return cleaned
        mode_upper = str(mode or "UNKNOWN").upper()
        if is_update_sensitive and mode_upper == "UPDATE" and not cleaned.startswith("[Update]"):
            return f"[Update] {cleaned}"
        if is_preference_sensitive and not cleaned.startswith("[Preference]"):
            return f"[Preference] {cleaned}"
        if status == "planned" and not cleaned.startswith("[Planned]"):
            return f"[Planned] {cleaned}"
        if status == "advice" and not cleaned.startswith("[Advice]"):
            return f"[Advice] {cleaned}"
        if status == "state" and not cleaned.startswith("[State]"):
            return f"[State] {cleaned}"
        return cleaned

    def _should_publish_to_generic_retrieval(
        self,
        should_archive: bool,
        mode: str,
        cluster_status: str,
        is_count_sensitive: bool,
        is_update_sensitive: bool,
        is_preference_sensitive: bool,
    ) -> bool:
        if should_archive:
            return True
        if cluster_status in {"advice", "planned"}:
            return False
        if is_count_sensitive or is_update_sensitive or is_preference_sensitive:
            return False
        if str(mode or "UNKNOWN").upper() == "AGGREGATE":
            return False
        return False

    def _should_submit_group_to_llm(self, facts: List[AtomicMemoryUnit]) -> Tuple[bool, str]:
        if len(facts) < self.MIN_CLUSTER_SIZE:
            return False, "too_small"

        speaker = self._infer_speaker_bucket(facts[0])
        status = self._infer_fact_status(facts[0])
        span_hours = self._estimate_time_span_hours(facts)
        unique_day_count = self._estimate_unique_day_count(facts)
        is_count_sensitive = self._is_count_sensitive_cluster(facts)
        is_update_sensitive = self._is_update_sensitive_cluster(facts)
        is_preference_sensitive = self._is_preference_sensitive_cluster(facts)

        # Keep count-sensitive facts separate to preserve event cardinality.

        if is_count_sensitive:
            return False, "skip_count_sensitive"

        # Exclude advice and plans from completed-event consolidation.
        if status in {"planned", "advice"}:
            return False, "skip_planned_or_advice"

        # preference 只保留用户侧、短小、状态型约束。
        if is_preference_sensitive:
            if speaker == "user" and status in {"state", "other"} and len(facts) <= 3 and unique_day_count <= 3:
                return True, "submit_preference"
            return False, "skip_preference_risky"

        # update 只保留很小的更新簇，目标是写出 current/latest。
        if is_update_sensitive:
            if len(facts) <= 2 and unique_day_count <= 2 and span_hours <= max(self.AGGREGATE_MAX_SPAN_HOURS, 72):
                return True, "submit_update"
            return False, "skip_update_risky"

        # Ordinary merging is limited to small, same-day user clusters.
        if speaker == "user" and len(facts) == 2 and unique_day_count <= 1 and span_hours <= 24:
            return True, "submit_small_user_merge"

        return False, "skip_low_value"

    def _build_group_prompt(
        self,
        context_text: str,
        speaker_scope: str,
        status_scope: str,
        earliest_ts: str,
        latest_ts: str,
        unique_day_count: int,
        count_sensitive: bool,
        update_sensitive: bool,
        preference_sensitive: bool
    ) -> str:
        if update_sensitive:
            return f"""
You are an expert Memory Analyst. Rewrite the facts into a tiny update record that is easy for a generic retriever to use.

[INPUT FACTS - Time Ordered]:
{context_text}

[GROUP PROFILE]:
- Speaker scope: {speaker_scope}
- Status scope: {status_scope}
- Earliest timestamp: {earliest_ts[:16]}
- Latest timestamp: {latest_ts[:16]}
- Unique calendar days: {unique_day_count}

[TASK]:
- Use UPDATE if a later fact changes or supersedes an earlier fact.
- Make the latest/current value explicit in the first fact.
- If useful, add one second fact for the previous value.
- Keep each fact short and self-contained.
- Preserve exact entities, dates, and numbers.
- Do not output generic summaries.

[OUTPUT JSON]:
{{
    "mode": "UPDATE" | "MERGE",
    "new_facts": ["Current/latest fact", "Optional previous fact"],
    "archived_indices": [0, 1]
}}
"""

        if preference_sensitive:
            return f"""
You are an expert Memory Analyst. Rewrite the facts into short user-preference constraints for future retrieval.

[INPUT FACTS - Time Ordered]:
{context_text}

[GROUP PROFILE]:
- Speaker scope: {speaker_scope}
- Status scope: {status_scope}
- Earliest timestamp: {earliest_ts[:16]}
- Latest timestamp: {latest_ts[:16]}
- Unique calendar days: {unique_day_count}

[TASK]:
- Capture the user's concrete likes, dislikes, exclusions, allergies, and preferred directions.
- Do not turn the output into assistant advice.
- Prefer wording like "User prefers...", "User wants...", "User is interested in...", "User is not interested in...".
- If the preference includes a comparison or exclusion, keep both sides explicit.
- Keep each fact short and self-contained.
- Preserve exact entities and exclusions.

[OUTPUT JSON]:
{{
    "mode": "MERGE" | "UPDATE",
    "new_facts": ["Preference fact 1"],
    "archived_indices": [0, 1]
}}
"""

        return f"""
You are an expert Memory Analyst. Merge only tightly related facts into a compact retrieval-friendly record.

[INPUT FACTS - Time Ordered]:
{context_text}

[GROUP PROFILE]:
- Speaker scope: {speaker_scope}
- Status scope: {status_scope}
- Earliest timestamp: {earliest_ts[:16]}
- Latest timestamp: {latest_ts[:16]}
- Unique calendar days: {unique_day_count}
- Count-sensitive cluster: {count_sensitive}

[TASK]:
- Prefer MERGE over AGGREGATE.
- Only combine facts if they describe the same actor and same situation.
- Keep the output short, self-contained, and precise.
- Preserve exact entities, dates, and numbers.
- Do not create high-level count summaries.

[OUTPUT JSON]:
{{
    "mode": "MERGE" | "UPDATE",
    "new_facts": ["Merged fact 1"],
    "archived_indices": [0, 1]
}}
"""

    def _get_archive_policy(self, facts: List[AtomicMemoryUnit], new_texts: List[str], mode: str) -> Tuple[bool, Dict[str, Any]]:
        concrete_speakers = {self._infer_speaker_bucket(fact) for fact in facts}
        has_neutral_inputs = "neutral" in concrete_speakers
        concrete_speakers.discard("neutral")

        span_hours = self._estimate_time_span_hours(facts)
        max_new_len = max((len(text or "") for text in new_texts), default=0)
        effective_mode = str(mode or "UNKNOWN").upper()

        should_archive = (
            effective_mode in {"MERGE", "UPDATE"}
            and len(facts) <= self.SAFE_ARCHIVE_MAX_PARENTS
            and span_hours <= self.SAFE_ARCHIVE_MAX_SPAN_HOURS
            and max_new_len <= self.SAFE_ARCHIVE_MAX_NEW_FACT_LEN
            and len(concrete_speakers) <= 1
            and not has_neutral_inputs
        )

        return should_archive, {
            "span_hours": span_hours,
            "speaker_scope": next(iter(concrete_speakers), "neutral"),
            "has_mixed_speakers": len(concrete_speakers) > 1 or has_neutral_inputs,
            "input_fact_count": len(facts),
            "archive_policy": "replace" if should_archive else "augment",
            "mode": effective_mode
        }

    def _rebuild_active_stats_snapshot(self):
        rebuilt = {
            "total_docs": 0,
            "code_freq": defaultdict(int),
            "co_occurrence": defaultdict(lambda: defaultdict(int))
        }
        if not os.path.exists(self.db_path):
            self.meta_db.stats = rebuilt
            return

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute("SELECT code_indices, metadata FROM atomic_memory")
            for code_indices_str, metadata_str in cur.fetchall():
                metadata = json.loads(metadata_str) if metadata_str else {}
                if metadata.get("status") == "archived":
                    continue

                codes = json.loads(code_indices_str) if code_indices_str else []
                unique_codes = sorted(set(codes))
                rebuilt["total_docs"] += 1

                for code in unique_codes:
                    rebuilt["code_freq"][code] += 1

                for idx, code_a in enumerate(unique_codes):
                    for code_b in unique_codes[idx + 1:]:
                        rebuilt["co_occurrence"][code_a][code_b] += 1
                        rebuilt["co_occurrence"][code_b][code_a] += 1
        finally:
            conn.close()

        self.meta_db.stats = rebuilt
        if hasattr(self.meta_db, "_save_stats"):
            self.meta_db._save_stats()

    # =========================================================================
    # [Step 1] 核心: 基于向量的层次聚类 (HAC)
    # =========================================================================
    async def _cluster_by_vector_semantics(self, facts: List[AtomicMemoryUnit]) -> List[List[AtomicMemoryUnit]]:
        """
        使用 Agglomerative Clustering 进行语义聚类。
        """
        valid_facts = [f for f in facts if f.embedding is not None and len(f.embedding) > 0]
        n_facts = len(valid_facts)
        if n_facts < self.MIN_CLUSTER_SIZE: return []

        embeddings = [f.embedding for f in valid_facts]
        X = np.array(embeddings)

        # 归一化
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        X_normalized = X / norms

        dist_threshold = 1.0 - self.SIMILARITY_THRESHOLD

        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage='average',
                distance_threshold=dist_threshold
            )
            model.fit(X_normalized)
            labels = model.labels_
        except Exception as e:
            print(f"  x [Clustering Error] {e}. Skipping batch.")
            return []

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(valid_facts[idx])

        final_groups = []
        for label, group in clusters.items():
            if len(group) < self.MIN_CLUSTER_SIZE: continue

            # 对超大 Cluster 进行切片
            if len(group) > self.MAX_CLUSTER_SIZE:
                group.sort(key=lambda x: str(x.metadata.get('timestamp', '')))
                for i in range(0, len(group), self.MAX_CLUSTER_SIZE):
                    chunk = group[i:i + self.MAX_CLUSTER_SIZE]
                    if len(chunk) >= self.MIN_CLUSTER_SIZE:
                        final_groups.append(chunk)
            else:
                final_groups.append(group)

        return final_groups

    # =========================================================================
    # [Main Flow]
    # =========================================================================
    async def execute_evolution_async(self):
        print("\n=== LongMemEval memory consolidation: speaker/status grouping and HAC ===")
        if hasattr(self.meta_db, "_save_stats"): self.meta_db._save_stats()

        stats_pre = self._get_detailed_db_stats()
        loop = asyncio.get_running_loop()

        # 1. 获取全量活跃事实
        raw_facts = await loop.run_in_executor(None, self.meta_db.get_all_active_atomics)
        if len(raw_facts) < self.MIN_CLUSTER_SIZE:
            print("  > 活跃事实不足，跳过。")
            return

        # 2. 向量回填
        print(f"  > 正在从 FAISS 回填 {len(raw_facts)} 条事实的向量...")
        all_ids = [f.id for f in raw_facts]
        try:
            vectors_map = await loop.run_in_executor(None, self.vec_db.get_vectors_by_ids, "atomic_memory", all_ids)
            hydrated_facts = []
            for f in raw_facts:
                if f.id in vectors_map:
                    f.embedding = vectors_map[f.id]
                    hydrated_facts.append(f)
        except Exception as e:
            hydrated_facts = []
            print(f"  x Hydration failed: {e}")

        if len(hydrated_facts) < self.MIN_CLUSTER_SIZE:
            print("  > 有效向量不足，跳过。")
            return

        # 3. 聚类: 先按 speaker + status 分桶，避免把 completed / planned / advice 混在一起
        narrative_groups: List[List[AtomicMemoryUnit]] = []
        submitted_groups: List[List[AtomicMemoryUnit]] = []
        skip_reason_counter: Counter = Counter()
        speaker_status_buckets = self._split_facts_by_speaker_and_status(hydrated_facts)
        for (speaker_bucket, status_bucket), bucket_facts in speaker_status_buckets.items():
            if len(bucket_facts) < self.MIN_CLUSTER_SIZE:
                continue
            if self._should_skip_group(bucket_facts):
                continue
            bucket_groups = await loop.run_in_executor(
                None,
                lambda current_bucket=bucket_facts: asyncio.run(
                    self._cluster_by_vector_semantics(current_bucket)
                )
            )
            filtered_groups = [group for group in bucket_groups if not self._should_skip_cluster(group)]
            narrative_groups.extend(filtered_groups)
            bucket_submitted = 0
            for group in filtered_groups:
                should_submit, reason = self._should_submit_group_to_llm(group)
                if should_submit:
                    submitted_groups.append(group)
                    bucket_submitted += 1
                else:
                    skip_reason_counter[reason] += 1
            print(
                f"  > Speaker bucket '{speaker_bucket}/{status_bucket}' generated "
                f"{len(filtered_groups)}/{len(bucket_groups)} groups after cluster-level filtering, "
                f"submitted {bucket_submitted}."
            )

        print(f"  > 聚类完成，生成了 {len(narrative_groups)} 个演化组。")
        print(f"  > 进入 LLM 演化的组数: {len(submitted_groups)}")
        if skip_reason_counter:
            print(f"  > 过滤原因统计: {dict(skip_reason_counter)}")

        global_stats = self.meta_db.get_stats()

        # 4. 并发执行
        tasks = []
        for group in submitted_groups:
            task = asyncio.create_task(self._process_narrative_group(group, global_stats))
            tasks.append(task)

        if not tasks: return

        print(f"  > 提交 {len(tasks)} 个演化任务...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)

        self._rebuild_active_stats_snapshot()
        stats_post = self._get_detailed_db_stats()
        print(
            f"=== 演化完成: 成功 {success_count}/{len(tasks)}, Active Change: {stats_post['total_active'] - stats_pre['total_active']} ===")

    # =========================================================================
    # [Core Logic] 叙事性演化
    # =========================================================================
    async def _process_narrative_group(self, facts: List[AtomicMemoryUnit], stats: Dict[str, Any]) -> bool:
        async with self.semaphore:
            try:
                # 按时间排序输入
                facts.sort(key=lambda x: str(x.metadata.get('timestamp', 'Unknown')))

                context_lines = []
                id_map = {}
                valid_timestamps = []
                speaker_buckets = set()
                status_buckets = set()

                # [Optimization] 精确追踪时间跨度
                for i, f in enumerate(facts):
                    raw_ts = str(f.metadata.get('timestamp', 'Unknown'))
                    if raw_ts and raw_ts != "Unknown":
                        valid_timestamps.append(raw_ts)

                    speaker_bucket = self._infer_speaker_bucket(f)
                    status_bucket = self._infer_fact_status(f)
                    speaker_buckets.add(speaker_bucket)
                    status_buckets.add(status_bucket)

                    # 仅展示 YYYY-MM-DD HH:MM
                    display_ts = raw_ts[:16]
                    context_lines.append(
                        f"ID[{i}] Speaker({speaker_bucket}) Status({status_bucket}) Time({display_ts}): {f.fact_text}"
                    )
                    id_map[i] = f

                # 计算最早和最晚时间
                if valid_timestamps:
                    valid_timestamps.sort()
                    earliest_ts = valid_timestamps[0]
                    latest_ts = valid_timestamps[-1]
                else:
                    earliest_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    latest_ts = earliest_ts

                context_text = "\n".join(context_lines)
                speaker_scope = ", ".join(sorted(speaker_buckets))
                status_scope = ", ".join(sorted(status_buckets))
                count_sensitive = self._is_count_sensitive_cluster(facts)
                update_sensitive = self._is_update_sensitive_cluster(facts)
                preference_sensitive = self._is_preference_sensitive_cluster(facts)
                unique_day_count = self._estimate_unique_day_count(facts)

                prompt = self._build_group_prompt(
                    context_text=context_text,
                    speaker_scope=speaker_scope,
                    status_scope=status_scope,
                    earliest_ts=earliest_ts,
                    latest_ts=latest_ts,
                    unique_day_count=unique_day_count,
                    count_sensitive=count_sensitive,
                    update_sensitive=update_sensitive,
                    preference_sensitive=preference_sensitive,
                )
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, lambda: self.llm.get_completion(prompt))
                result = self._parse_json(response)

                new_texts = result.get("new_facts", [])
                archived_indices = result.get("archived_indices", [])
                mode = result.get("mode", "UNKNOWN")

                if not new_texts or not archived_indices: return False
                if len(new_texts) >= len(archived_indices): return False

                await loop.run_in_executor(
                    None,
                    self._perform_updates_sync,
                    new_texts, archived_indices, id_map, latest_ts, earliest_ts, stats, mode
                )
                return True

            except Exception as e:
                return False

    def _perform_updates_sync(self, new_texts, archived_indices, id_map, latest_ts, earliest_ts, stats, mode):
        """
        LongMemEval 安全策略:
        1. 高风险聚合只增广，不替换原子事实
        2. 时间跨度保留在 metadata，不注入到 fact_text
        """
        with self.write_lock:
            local_meta_db = MetadataDBHandler(self.db_path)
            try:
                # 1. 预先生成新事实 ID
                new_fact_ids = [str(uuid.uuid4()) for _ in new_texts]

                selected_mems = []
                old_ids = []
                parent_codes = []
                source_base_ids_pool = set()

                for idx in archived_indices:
                    if idx not in id_map:
                        continue
                    mem = id_map[idx]
                    selected_mems.append(mem)
                    old_ids.append(mem.id)
                    parent_codes.extend(mem.code_indices)
                    if mem.source_base_ids:
                        source_base_ids_pool.update(mem.source_base_ids)

                if not selected_mems or not new_texts:
                    return

                should_archive, archive_meta = self._get_archive_policy(selected_mems, new_texts, mode)
                fact_statuses = {self._infer_fact_status(mem) for mem in selected_mems}
                cluster_status = next(iter(fact_statuses), "other") if len(fact_statuses) == 1 else "mixed"
                is_count_sensitive = self._is_count_sensitive_cluster(selected_mems)
                is_update_sensitive = self._is_update_sensitive_cluster(selected_mems)
                is_preference_sensitive = self._is_preference_sensitive_cluster(selected_mems)
                unique_day_count = self._estimate_unique_day_count(selected_mems)
                mode_upper = str(mode).upper()

                # Reject a single summary that would collapse multiple countable events.


                if (
                    str(mode).upper() == "AGGREGATE"
                    and is_count_sensitive
                    and len(selected_mems) >= 3
                    and len(new_texts) <= 1
                ):
                    return

                # Do not write aggregate advice as a factual event.
                if str(mode).upper() == "AGGREGATE" and cluster_status == "advice":
                    return

                if (
                    mode_upper == "AGGREGATE"
                    and is_count_sensitive
                    and cluster_status == "completed"
                    and (unique_day_count > 1 or len(selected_mems) > 2)
                ):
                    return

                if is_update_sensitive:
                    if mode_upper != "UPDATE":
                        return
                    update_markers = ("latest", "current", "most recent", "now", "updated", "personal best")
                    if not any(any(marker in str(text).lower() for marker in update_markers) for text in new_texts):
                        return

                if is_preference_sensitive:
                    if cluster_status not in {"state", "other"}:
                        return
                    preference_markers = (
                        "prefer", "favorite", "like", "dislike", "interested", "allergic", "not interested",
                        "looking for", "similar to", "instead of", "rather than", "beyond"
                    )
                    if not any(any(marker in str(text).lower() for marker in preference_markers) for text in new_texts):
                        return

                if should_archive:
                    for mem in selected_mems:
                        local_meta_db.archive_atomic_memory(
                            mem.id,
                            reason=f"Evolved ({mode})",
                            extra_updates={'superseded_by': new_fact_ids}
                        )

                archived_mems = selected_mems

                try:
                    new_embeddings = self.embedder.embed_batch(new_texts)
                except:
                    new_embeddings = [self.embedder.embed(t) for t in new_texts]

                # Codebook indices handling
                new_native_codes_list = [[] for _ in new_texts]
                if self.vq_model:
                    try:
                        new_native_codes_list = self.vq_model.get_indices(new_texts)
                    except:
                        pass

                final_source_ids = list(source_base_ids_pool)
                if len(final_source_ids) > 10: final_source_ids = final_source_ids[-10:]

                new_objects = []
                vec_data = []

                # 时间处理
                final_ts = latest_ts if latest_ts else datetime.now().isoformat()
                final_start_ts = earliest_ts if earliest_ts else final_ts

                for i, text in enumerate(new_texts):
                    # Limit inherited codes for supplemental summaries.

                    final_parent_codes = set()
                    code_counts = Counter(parent_codes)
                    publish_to_generic = self._should_publish_to_generic_retrieval(
                        should_archive=should_archive,
                        mode=mode_upper,
                        cluster_status=cluster_status,
                        is_count_sensitive=is_count_sensitive,
                        is_update_sensitive=is_update_sensitive,
                        is_preference_sensitive=is_preference_sensitive,
                    )
                    if cluster_status in {"advice", "planned"} or not publish_to_generic:
                        final_parent_codes = set()
                    elif mode_upper == "AGGREGATE":
                        if should_archive:
                            for c, count in code_counts.items():
                                if self._calculate_idf(c, stats) > 1.0:
                                    final_parent_codes.add(c)
                        else:
                            num_parents = max(len(archived_mems), 1)
                            for c, count in code_counts.items():
                                if count / num_parents >= 0.8 and self._calculate_idf(c, stats) > 1.3:
                                    final_parent_codes.add(c)
                    else:
                        num_parents = len(archived_mems)
                        for c, count in code_counts.items():
                            if num_parents > 0 and (count / num_parents >= 0.4):
                                final_parent_codes.add(c)

                    final_codes = list(set(new_native_codes_list[i]) | final_parent_codes) if publish_to_generic else []
                    new_id = new_fact_ids[i]
                    final_text = self._format_fact_text_for_old_retrieval(
                        text,
                        cluster_status,
                        mode=mode_upper,
                        is_update_sensitive=is_update_sensitive,
                        is_preference_sensitive=is_preference_sensitive,
                    )

                    new_mem = AtomicMemoryUnit(
                        id=new_id,
                        fact_text=final_text,
                        source_base_ids=final_source_ids,
                        embedding=new_embeddings[i],
                        code_indices=final_codes,
                        metadata={
                            'type': 'evolved_fact',
                            'timestamp': final_ts,
                            'timestamp_start': final_start_ts,
                            'timestamp_end': final_ts,
                            'evolution_mode': mode,
                            'created_at': datetime.now().isoformat(),
                            'derived_from': old_ids,
                            'derived_from_count': len(old_ids),
                            'archive_policy': archive_meta['archive_policy'],
                            'speaker_scope': archive_meta['speaker_scope'],
                            'fact_status': cluster_status,
                            'has_mixed_speakers': archive_meta['has_mixed_speakers'],
                            'temporal_span_hours': archive_meta['span_hours'],
                            'retrieval_priority': 'high' if should_archive else 'supplemental',
                            'is_aggregate_summary': str(mode).upper() == 'AGGREGATE',
                            'count_risk': (str(mode).upper() == 'AGGREGATE' and not should_archive),
                            'count_sensitive_cluster': is_count_sensitive,
                            'update_sensitive_cluster': is_update_sensitive,
                            'preference_sensitive_cluster': is_preference_sensitive,
                            'generic_retrieval_visible': publish_to_generic
                        }
                    )
                    local_meta_db.add_atomic_memory(new_mem)
                    new_objects.append(new_mem)
                    if publish_to_generic and new_mem.embedding:
                        vec_data.append((new_id, new_mem.embedding))

                if vec_data:
                    self.vec_db.add_vectors_batch("atomic_memory", vec_data)

                local_meta_db.update_stats(new_objects)

            except Exception as e:
                print(f"  x [Write Error] {e}")
            finally:
                local_meta_db.close()

    def _parse_json(self, text):
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            return json.loads(cleaned)
        except:
            return {}
