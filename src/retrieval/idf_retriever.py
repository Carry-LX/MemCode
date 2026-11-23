import math
from typing import List, Any, Dict, Set, Tuple
from collections import defaultdict

import concurrent.futures

# 假设引用路径保持不变
from src.utils.vqvae_inference_v10 import ModelInference
from src.core.memory_structures import BaseMemoryUnit, AtomicMemoryUnit
from src.core.metadata_db_handler import MetadataDBHandler
from src.core.vector_db_handler import VectorDBHandler
from src.utils.embedding_bge import EmbeddingModel


class QueryEngine:
    """Retrieval ablation with IDF weighting and low-IDF filtering disabled.

    All query-code seed weights are 1.0. Dense-vector retrieval and
    co-occurrence neighbor expansion remain enabled."""

    # Hyperparameters
    ALPHA_DECAY: float = 0.3
    EXPANSION_K: int = 3

    # [失效] 阈值在这里不再起作用，因为我们不计算 IDF
    MIN_IDF_THRESHOLD: float = 0.5

    # 保持原版阈值，控制变量
    MAX_VECTOR_DISTANCE: float = 0.65
    MIN_CODEBOOK_SCORE: float = 0.3

    def __init__(self,
                 embedder: EmbeddingModel,
                 meta_db: MetadataDBHandler,
                 vec_db: VectorDBHandler,
                 llm_interface: Any,
                 vq_model=None,
                 stop_ratio: float = 0.15):

        self.embedder = embedder
        self.meta_db = meta_db
        self.vec_db = vec_db
        self.llm = llm_interface

        if vq_model:
            self.vq_model = vq_model
        else:
            self.vq_model = ModelInference(device_str='cuda')

        print("⚠️ [Ablation Experiment] Query Engine initialized with UNIFORM WEIGHTING (No IDF).")

    # =========================================================================
    # Ablation: replace IDF weights with a constant.
    # =========================================================================

    def _calculate_idf(self, code: int, stats: Dict[str, Any]) -> float:
        """Return a constant seed weight of 1.0 for this ablation."""
        return 1.0

    def _get_weighted_neighbors(self, source_code: int, stats: Dict[str, Any], top_k: int = 3) -> List[
        Tuple[int, float]]:
        """Return co-occurrence neighbors and transition probabilities.

        The query-code seed weight is constant rather than IDF-weighted."""
        graph = stats.get("co_occurrence", {})
        neighbors_map = graph.get(source_code, {})
        if not neighbors_map: return []

        source_freq = stats.get("code_freq", {}).get(source_code, 1)

        candidates = []
        for n_code, co_count in neighbors_map.items():
            # 计算转移概率 P(B|A)
            # 注意：高频词 source_freq 很大，所以单个邻居的概率可能很小，
            # High-frequency codes remain eligible for neighbor expansion.
            prob = co_count / (source_freq + 1)
            candidates.append((n_code, prob))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # =========================================================================
    # 检索逻辑
    # =========================================================================
    def _retrieve_atomic_candidates(self, queries: List[str], k_per_step: int, total_atomic_count: int = None) -> List[
        AtomicMemoryUnit]:
        final_candidates_map: Dict[str, AtomicMemoryUnit] = {}
        seen_ids: Set[str] = set()

        current_stats = self.meta_db.get_stats()

        # 1. Embedding & VQ Encoding
        query_embeddings = []
        query_codes_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_embed = executor.submit(self.embedder.embed, queries, is_query=True)
            if self.vq_model:
                future_vq = executor.submit(self.vq_model.get_indices, queries)

            try:
                query_embeddings = future_embed.result()
                if self.vq_model:
                    query_codes_list = future_vq.result()
            except:
                pass

        current_embedding = query_embeddings[0] if len(query_embeddings) > 0 else None
        current_codes = query_codes_list[0] if len(query_codes_list) > 0 else []

        search_tokens: Dict[int, float] = {}

        # Path A: Vector Search (Control Group - Kept Intact)
        if current_embedding is not None:
            vec_results = self.vec_db.search("atomic_memory", current_embedding, k=k_per_step * 2)
            for amu_id, dist in vec_results:
                if len(final_candidates_map) >= k_per_step: break
                if amu_id not in seen_ids:
                    mem = self.meta_db.get_atomic_memory_by_id(amu_id)
                    if mem:
                        if mem.metadata is None: mem.metadata = {}
                        mem.metadata['vector_score'] = float(dist)
                        final_candidates_map[amu_id] = mem
                        seen_ids.add(amu_id)

        # Path B: Codebook Search (Target Group - IDF Removed)
        if current_codes and current_stats.get("total_docs", 0) > 0:

            # Step 1: 种子编码处理
            for q_code in current_codes:
                # 强制权重为 1.0 (由 _calculate_idf 返回)
                base_weight = self._calculate_idf(q_code, current_stats)

                # 移除 IDF 阈值过滤
                # 消融逻辑: 无论是什么词，通通放进来。
                # All query codes remain eligible as search seeds.

                search_tokens[q_code] = max(search_tokens.get(q_code, 0), base_weight)

                # Step 2: graph expansion using constant seed weights.
                # Expand neighbors without low-IDF filtering.
                neighbors = self._get_weighted_neighbors(q_code, current_stats, top_k=self.EXPANSION_K)

                for n_code, trans_prob in neighbors:
                    # n_weight = 1.0 * prob * 0.3
                    n_weight = base_weight * trans_prob * self.ALPHA_DECAY

                    if n_weight > search_tokens.get(n_code, 0):
                        search_tokens[n_code] = n_weight

            # Step 3: 累分
            candidate_scores: Dict[str, float] = defaultdict(float)
            for code, weight in search_tokens.items():
                atomic_ids = self.meta_db.get_atomic_ids_by_code(code)
                for aid in atomic_ids:
                    candidate_scores[aid] += weight

            # Step 4: 排序与选取
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

            path_b_count = 0
            needed = k_per_step

            for amu_id, score in sorted_candidates:
                if path_b_count >= needed: break
                if amu_id not in seen_ids:
                    mem = self.meta_db.get_atomic_memory_by_id(amu_id)
                    if mem:
                        if mem.metadata is None: mem.metadata = {}
                        mem.metadata['vector_score'] = 999.0  # 标记为来自 Codebook 路径
                        final_candidates_map[amu_id] = mem
                        seen_ids.add(amu_id)
                        path_b_count += 1

        # 全员回填 Codebook Score
        # 注意：这里的 score 计算也是基于无权重的 search_tokens
        for amu_id, mem in final_candidates_map.items():
            fact_codes = getattr(mem, 'code_indices', []) or []
            recalculated_score = 0.0
            for fc in fact_codes:
                if fc in search_tokens:
                    recalculated_score += search_tokens[fc]

            mem.metadata['codebook_score'] = recalculated_score
            if 'vector_score' not in mem.metadata:
                mem.metadata['vector_score'] = 999.0

        return list(final_candidates_map.values())

    def _group_and_resolve_to_base(self, atomics: List[AtomicMemoryUnit]) -> Tuple[
        List[BaseMemoryUnit], Dict[str, List[AtomicMemoryUnit]]]:
        base_map: Dict[str, BaseMemoryUnit] = {}
        base_to_atomics_map = defaultdict(list)

        for amu in atomics:
            if not amu.source_base_ids: continue
            for base_id in amu.source_base_ids:
                base_to_atomics_map[base_id].append(amu)
                if base_id not in base_map:
                    base_mem = self.meta_db.get_base_memory_by_id(base_id)
                    if base_mem:
                        base_map[base_id] = base_mem

        ordered_unique_bases = []
        added_base_ids = set()

        for amu in atomics:
            v_score = amu.metadata.get('vector_score', 999.0)
            c_score = amu.metadata.get('codebook_score', 0.0)

            # Apply the stored evidence thresholds to unweighted code-path scores.
            is_qualified = (v_score < self.MAX_VECTOR_DISTANCE) or (c_score > self.MIN_CODEBOOK_SCORE)

            if is_qualified:
                for base_id in amu.source_base_ids:
                    if base_id in base_map and base_id not in added_base_ids:
                        ordered_unique_bases.append(base_map[base_id])
                        added_base_ids.add(base_id)

        return ordered_unique_bases, base_to_atomics_map

    # 主入口保持不变
    def query(self, question: str, k: int = 5, total_atomic_count: int = None, query_date: str = None, **kwargs) -> \
    Dict[str, Any]:

        print(f"\n--- Querying (IDF Ablation Mode): '{question}' ---")


        trace_info = {
            "question": question,
            "strategies_used": ["uniform_weighting", "graph_expansion"],
            "retrieved_pool_size": 0
        }
        queries = [question]

        # 检索
        raw_atomics = self._retrieve_atomic_candidates(queries, k_per_step=20, total_atomic_count=total_atomic_count)
        trace_info["retrieved_pool_size"] = len(raw_atomics)

        # 映射
        unique_bases, _ = self._group_and_resolve_to_base(raw_atomics)
        selected_bases = unique_bases[:25]
        final_atomics = raw_atomics[:40]

        # 构造 Prompt
        context_str = self._format_evidence(final_atomics, selected_bases)
        final_prompt = self._build_final_prompt(question, context_str, query_date=query_date)

        return {
            "final_prompt": final_prompt,
            "trace_info": trace_info
        }

    def _format_evidence(self, atomics, bases):

        text = "[EVIDENCE FACTS]:\n"
        for i, f in enumerate(atomics):
            text += f"{i + 1}. {f.fact_text}\n"
        text += "\n[SOURCE CONTEXT]:\n"
        bases.sort(key=lambda x: str(x.timestamp) if x.timestamp else "")
        for b in bases:
            text += f"- {b.metadata.get('speaker', 'Unknown')}: {b.text}\n"
        return text

    def _build_final_prompt(self, question, evidence, query_date=None):

        return f"Question: {question}\nContext:\n{evidence}"
