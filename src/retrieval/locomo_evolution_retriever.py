import json
import math
import re
from typing import List, Any, Dict, Set, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import concurrent.futures

from src.utils.vqvae_inference_v10 import ModelInference
from src.core.memory_structures import BaseMemoryUnit, AtomicMemoryUnit
from src.core.metadata_db_handler import MetadataDBHandler
from src.core.vector_db_handler import VectorDBHandler
from src.utils.embedding_bge import EmbeddingModel


class QueryEngine:
    """LoCoMo retrieval with archived-to-active fact redirection.

    Dense-vector and codebook paths have separate candidate quotas.
    Resolved evolved facts preserve links to their source dialogue units."""

    # [Hyperparameters] 超参数定义
    ALPHA_DECAY: float = 0.3
    EXPANSION_K: int = 2
    MIN_IDF_THRESHOLD: float = 0.5
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
        self.stop_ratio = stop_ratio
        self.ENABLE_DYNAMIC_THRESHOLD = False

        if vq_model:
            self.vq_model = vq_model
        else:
            self.vq_model = ModelInference(device_str='cuda')

        if not self.vq_model:
            print("⚠️ VQ-VAE 模型加载失败，请检查模型路径。")

        print("智能查询引擎 (Redirect Enhanced) 已初始化。")

    # =========================================================================
    # 数学辅助函数
    # =========================================================================
    def _calculate_idf(self, code: int, stats: Dict[str, Any]) -> float:
        total_docs = stats.get("total_docs", 0)
        if total_docs == 0: return 0.0
        freq_map = stats.get("code_freq", {})
        doc_freq = freq_map.get(code, 0)
        idf = math.log(total_docs / (doc_freq + 1))
        return max(0.0, idf)

    def _get_weighted_neighbors(self, source_code: int, stats: Dict[str, Any], top_k: int = 3) -> List[
        Tuple[int, float]]:
        graph = stats.get("co_occurrence", {})
        neighbors_map = graph.get(source_code, {})
        if not neighbors_map: return []
        source_freq = stats.get("code_freq", {}).get(source_code, 1)

        candidates = []
        for n_code, co_count in neighbors_map.items():
            prob = co_count / (source_freq + 1)
            candidates.append((n_code, prob))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # =========================================================================
    # 策略 2: 初筛 (Retrieval) - 找原子事实 (含重定向逻辑)
    # =========================================================================
    def _retrieve_atomic_candidates(self, queries: List[str], k_per_step: int, total_atomic_count: int = None) -> List[
        AtomicMemoryUnit]:
        """
        执行多样性检索，返回候选的 AtomicMemoryUnit 列表。
        """
        final_candidates_map: Dict[str, AtomicMemoryUnit] = {}
        seen_ids: Set[str] = set()

        current_stats = self.meta_db.get_stats()

        # 并行执行模型推理
        query_embeddings = []
        query_codes_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_embed = executor.submit(self.embedder.embed, queries, is_query=True)
            future_vq = None
            if self.vq_model:
                future_vq = executor.submit(self.vq_model.get_indices, queries)
            try:
                query_embeddings = future_embed.result()
            except Exception:
                query_embeddings = []
            if future_vq:
                try:
                    query_codes_list = future_vq.result()
                except:
                    query_codes_list = []

        current_embedding = query_embeddings[0] if len(query_embeddings) > 0 else None
        current_codes = query_codes_list[0] if len(query_codes_list) > 0 else []

        search_tokens: Dict[int, float] = {}


        # Helper: 处理单个候选 (包含重定向逻辑)
        # 返回: (Added_Flag, Mem_Object)

        def try_resolve_candidate(mem_id: str) -> AtomicMemoryUnit:
            mem = self.meta_db.get_atomic_memory_by_id(mem_id)
            if not mem: return None

            status = mem.metadata.get('status', 'active')

            # === [核心逻辑: 重定向] ===
            if status == 'archived':
                superseded_list = mem.metadata.get('superseded_by', [])
                if isinstance(superseded_list, list) and len(superseded_list) > 0:
                    for new_id in superseded_list:
                        # 递归找回 active 事实
                        new_mem = self.meta_db.get_atomic_memory_by_id(new_id)
                        # 确保新事实未被归档
                        if new_mem and new_mem.metadata.get('status', 'active') != 'archived':
                            if new_mem.metadata is None: new_mem.metadata = {}
                            new_mem.metadata['redirected_from'] = mem_id
                            return new_mem
                return None  # 无法重定向或本身就是归档

            return mem

        # Path A: Vector Search (语义主路)
        if current_embedding is not None:
            # 扩大搜索范围 (k*3)，因为归档数据会占用坑位，我们需要多拿一些来重定向
            vec_results = self.vec_db.search("atomic_memory", current_embedding, k=k_per_step * 3)

            for amu_id, dist in vec_results:
                # [Strict Logic]: 向量只能拿 k 个
                if len(final_candidates_map) >= k_per_step: break

                if amu_id in seen_ids: continue

                # 尝试解析 (含重定向)
                mem = try_resolve_candidate(amu_id)

                if mem and mem.id not in seen_ids:
                    if mem.metadata is None: mem.metadata = {}
                    mem.metadata['vector_score'] = float(dist)

                    final_candidates_map[mem.id] = mem
                    seen_ids.add(mem.id)

        # Path B: Structure-Enhanced Codebook Search (联想支路)
        if current_codes and current_stats.get("total_docs", 0) > 0:
            for q_code in current_codes:
                idf = self._calculate_idf(q_code, current_stats)
                if idf < self.MIN_IDF_THRESHOLD: continue
                search_tokens[q_code] = max(search_tokens.get(q_code, 0), idf)
                neighbors = self._get_weighted_neighbors(q_code, current_stats, top_k=self.EXPANSION_K)
                for n_code, trans_prob in neighbors:
                    n_weight = idf * trans_prob * self.ALPHA_DECAY
                    if n_weight > search_tokens.get(n_code, 0):
                        search_tokens[n_code] = n_weight

            candidate_scores: Dict[str, float] = defaultdict(float)
            for code, weight in search_tokens.items():
                atomic_ids = self.meta_db.get_atomic_ids_by_code(code)
                for aid in atomic_ids:
                    candidate_scores[aid] += weight

            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

            # [Strict Logic]: 码本补充 k 个
            path_b_count = 0
            needed = k_per_step

            for amu_id, score in sorted_candidates:
                if path_b_count >= needed: break

                if amu_id in seen_ids: continue

                # 尝试解析 (含重定向)
                mem = try_resolve_candidate(amu_id)

                # 只有真正加入了一个不在池子里的新事实，才算消耗了配额
                if mem and mem.id not in seen_ids:
                    if mem.metadata is None: mem.metadata = {}
                    mem.metadata['codebook_score'] = score
                    if 'vector_score' not in mem.metadata:
                        mem.metadata['vector_score'] = 999.0

                    final_candidates_map[mem.id] = mem
                    seen_ids.add(mem.id)
                    path_b_count += 1

        # 全员普查：重新计算 codebook_score
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

            is_redirected = 'redirected_from' in amu.metadata
            is_evolved = amu.metadata.get('type') == 'evolved_fact'

            is_qualified = (v_score < self.MAX_VECTOR_DISTANCE) or (c_score > self.MIN_CODEBOOK_SCORE)

            # 只要是演化事实或重定向事实，默认可信
            if is_redirected or is_evolved:
                is_qualified = True

            if is_qualified:
                for base_id in amu.source_base_ids:
                    if base_id in base_map and base_id not in added_base_ids:
                        ordered_unique_bases.append(base_map[base_id])
                        added_base_ids.add(base_id)

        return ordered_unique_bases, base_to_atomics_map

    # =========================================================================
    # 主查询入口
    # =========================================================================
    def query(self, question: str, k: int = 5, total_atomic_count: int = None, query_date: str = None, **kwargs) -> \
            Dict[str, Any]:
        print(f"\n--- Querying (Efficiency Optimized): '{question}' ---")

        trace_info = {
            "question": question,
            "strategies_used": ["direct_retrieval", "codebook_graph_expansion", "idf_weighted_scoring"],
            "expanded_queries": [],
            "retrieved_pool_size": 0
        }

        queries = [question]
        trace_info["expanded_queries"] = queries

        # 初筛
        k = 20
        raw_atomics = self._retrieve_atomic_candidates(queries, k_per_step=k, total_atomic_count=total_atomic_count)
        trace_info["retrieved_pool_size"] = len(raw_atomics)
        print(f"  - 检索到的原子事实数量为:  {len(raw_atomics)} 条.")

        # 映射与截断
        unique_bases, _ = self._group_and_resolve_to_base(raw_atomics)
        selected_bases = unique_bases[:10]
        final_atomics = raw_atomics[:40]
        print(f"  - [Selection] Truncated top {len(selected_bases)} 原始对话.")
        print(f"  - [Selection] Truncated top {len(final_atomics)} 原子事实.")

        context_str = self._format_evidence(final_atomics, selected_bases)
        final_prompt = self._build_final_prompt(question, context_str, query_date=query_date)

        return {
            "final_prompt": final_prompt,
            "trace_info": trace_info
        }

    def _format_evidence(self, atomics: List[AtomicMemoryUnit], bases: List[BaseMemoryUnit]) -> str:
        text = "[EVIDENCE FACTS (Abstracted Insights)]:\n"
        if not atomics: text += "No specific facts extracted.\n"
        for i, f in enumerate(atomics):
            ts_str = ""
            if f.metadata.get('timestamp'):
                ts_str = f"[Message Date: {str(f.metadata.get('timestamp'))[:16]}] "
            text += f"{i + 1}. {f.fact_text}{ts_str}\n"

        text += "\n[SOURCE CONVERSATION (Original Context)]:\n"
        bases.sort(key=lambda x: str(x.timestamp) if x.timestamp else "")
        for b in bases:
            speaker = b.metadata.get('speaker', 'Unknown')
            ts = str(b.timestamp)[:16] if b.timestamp else ""
            content = b.text
            text += f"- [{ts}] {speaker}: {content}\n"

        text += "\n[INSTRUCTIONS FOR EVIDENCE PROCESSING]:\n"
        text += "1. **Prioritize Evidence & Completeness**: Your answer must be rooted in the situations described in the [EVIDENCE CONTEXT]. If the question asks for a list or multiple items, ensure you mention **all** relevant items found in the context, do not omit minor details.\n"
        text += "2. **Look for Exact Details**: The [SOURCE CONVERSATION] section contains raw details. Use the Source Context to find exact answers (e.g., use specific names like \"banana split\" instead of general terms like \"dessert\"). Do not generalize specific entities.\n"
        text += "3. **Timeline & Calculation**: The timestamps are in YYYY-MM-DD HH:MM format. Pay attention to the **Time** to distinguish events. If the text mentions relative time (e.g., \"two weeks ago\"), calculate the actual timing based on the message's timestamp.\n"
        text += "4. **Reasoning**: Combine facts from different parts of the conversation to form a complete answer. If specific preferences are mentioned, include them.\n"
        return text

    def _build_final_prompt(self, question: str, evidence: str, query_date: str = None) -> str:
        time_instruction = ""
        if query_date:
            print(f"[Query Date] {query_date}")
            time_instruction = f"""
            [CURRENT REFERENCE TIME]
            Current Date: {query_date}
            (Instruction: Assume 'today' or 'now' refers to this specific date. Calculate relative time based on this reference.)
            """

        return f"""
                You are an expert assistant. Answer the user's question using the provided evidence.
                {time_instruction}

                [USER QUESTION]
                {question}

                [EVIDENCE CONTEXT]
                {evidence}

                [YOUR ANSWER]
                """
