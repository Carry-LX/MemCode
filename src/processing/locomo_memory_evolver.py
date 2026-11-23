import json
import os
import asyncio
import uuid
import threading
import sqlite3
import numpy as np
import math
from typing import List, Dict, Any, Set
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
    """Asynchronous LoCoMo memory consolidation.

    Cluster active facts by vector similarity, retain source pointers and
    redirect archived parents through superseded_by. Consolidated text
    includes an explicit time span when parent timestamps differ."""

    # 超参数配置
    SIMILARITY_THRESHOLD = 0.80
    MIN_CLUSTER_SIZE = 2
    MAX_CLUSTER_SIZE = 50

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

        print("  > [Config] LoCoMo consolidation: HAC, source pointers and text time spans")

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
        print("\n=== LoCoMo memory consolidation: HAC and source pointers ===")
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

        # 3. 聚类
        narrative_groups = await loop.run_in_executor(None, lambda: asyncio.run(
            self._cluster_by_vector_semantics(hydrated_facts)))

        print(f"  > 聚类完成，生成了 {len(narrative_groups)} 个演化组。")

        global_stats = self.meta_db.get_stats()

        # 4. 并发执行
        tasks = []
        for group in narrative_groups:
            task = asyncio.create_task(self._process_narrative_group(group, global_stats))
            tasks.append(task)

        if not tasks: return

        print(f"  > 提交 {len(tasks)} 个演化任务...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)

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

                # [Optimization] 精确追踪时间跨度
                for i, f in enumerate(facts):
                    raw_ts = str(f.metadata.get('timestamp', 'Unknown'))
                    if raw_ts and raw_ts != "Unknown":
                        valid_timestamps.append(raw_ts)

                    # 仅展示 YYYY-MM-DD HH:MM
                    display_ts = raw_ts[:16]
                    context_lines.append(f"ID[{i}] Time({display_ts}): {f.fact_text}")
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

                prompt = f"""
You are an expert Memory Analyst. Your goal is to synthesize multiple atomic facts into a comprehensive knowledge record.

[INPUT FACTS - Time Ordered]:
{context_text}

[ANALYSIS INSTRUCTIONS]:
Determine the relationship and synthesize using ONE mode:

1. **AGGREGATE**: For distinct events of the same type.
   - **CRITICAL**: You MUST explicitly mention the specific date/time for EACH sub-event in the text.
   - Bad: "User went to the gym twice."
   - Good: "User went to the gym twice: once on 2023-05-01 and again on 2023-05-05."

2. **MERGE**: For different details of the same object/event.
   - Combine into one high-density sentence.

3. **UPDATE**: When a later fact invalidates an earlier one.
   - Report the final state.

[CONSTRAINTS]:
- **Time Preservation**: Never lose a timestamp. If condensing, include the range or specific instances.
- **Completeness**: Retain all entities and numbers.
- **Format**: JSON only.

[OUTPUT JSON]:
{{
    "mode": "AGGREGATE" | "MERGE" | "UPDATE",
    "new_facts": ["Synthesized Fact 1"],
    "archived_indices": [0, 1, 2]
}}
"""
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
        with self.write_lock:
            local_meta_db = MetadataDBHandler(self.db_path)
            try:
                # 1. 预先生成新事实 ID
                new_fact_ids = [str(uuid.uuid4()) for _ in new_texts]

                # 2. 归档旧事实 (建立 Pointer 链接)
                archived_mems = []
                old_ids = []
                parent_codes = []
                source_base_ids_pool = set()

                for idx in archived_indices:
                    if idx in id_map:
                        mem = id_map[idx]
                        archived_mems.append(mem)
                        old_ids.append(mem.id)
                        parent_codes.extend(mem.code_indices)
                        if mem.source_base_ids:
                            source_base_ids_pool.update(mem.source_base_ids)

                        # 写入 superseded_by
                        local_meta_db.archive_atomic_memory(
                            mem.id,
                            reason=f"Evolved ({mode})",
                            extra_updates={'superseded_by': new_fact_ids}
                        )

                if not new_texts: return

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

                # Code inheritance logic
                final_parent_codes = set()
                code_counts = Counter(parent_codes)
                if mode == "AGGREGATE":
                    for c, count in code_counts.items():
                        if self._calculate_idf(c, stats) > 1.0:
                            final_parent_codes.add(c)
                else:
                    num_parents = len(archived_mems)
                    for c, count in code_counts.items():
                        if num_parents > 0 and (count / num_parents >= 0.4):
                            final_parent_codes.add(c)

                final_source_ids = list(source_base_ids_pool)
                if len(final_source_ids) > 10: final_source_ids = final_source_ids[-10:]

                new_objects = []
                vec_data = []

                # 时间处理
                final_ts = latest_ts if latest_ts else datetime.now().isoformat()
                final_start_ts = earliest_ts if earliest_ts else final_ts

                # 仅当时间跨度显著不同（跨度超过分钟级）时，才注入 Span 标记。
                # 这样即使检索器只显示末尾的 [Message Date]，LLM 也能从内容开头看到 [Time Span]。
                should_inject_span = False
                if final_start_ts[:16] != final_ts[:16]:
                    should_inject_span = True

                # 格式化 Span 字符串
                span_str = f"[Time Span: {final_start_ts[:16]} to {final_ts[:16]}] "

                for i, text in enumerate(new_texts):
                    final_codes = list(set(new_native_codes_list[i]) | final_parent_codes)
                    new_id = new_fact_ids[i]

                    # 如果检测到时间跨度，强制修改文本内容
                    final_text = text
                    if should_inject_span and not text.startswith("[Time Span"):
                        final_text = span_str + text

                    new_mem = AtomicMemoryUnit(
                        id=new_id,
                        fact_text=final_text,  # 使用注入了时间的文本
                        source_base_ids=final_source_ids,
                        embedding=new_embeddings[i],
                        code_indices=final_codes,
                        metadata={
                            'type': 'evolved_fact',
                            'timestamp': final_ts,  # 保持最新时间用于排序/Recency
                            'timestamp_start': final_start_ts,  # 冗余存储，供未来分析
                            'timestamp_end': final_ts,
                            'evolution_mode': mode,
                            'created_at': datetime.now().isoformat(),
                            'derived_from': old_ids
                        }
                    )
                    local_meta_db.add_atomic_memory(new_mem)
                    new_objects.append(new_mem)
                    if new_mem.embedding:
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
