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
    """Dense-only candidate-selection variant with code-score annotations.

    The query entry requests up to 20 vector candidates and adds no new
    code-path candidates (needed = 0)."""

    # [Hyperparameters] 超参数定义
    # 衰减系数: 控制扩展邻居的权重 (0.3 表示邻居权重最大只有原词的 30%)
    ALPHA_DECAY: float = 0.3

    # 扩展数量: 每个 Query Code 最多扩展多少个强关联邻居
    EXPANSION_K: int = 3

    # 最小 IDF 阈值: 忽略 IDF 低于此值的 Query Code (过滤 "is", "the" 等停用语义)
    MIN_IDF_THRESHOLD: float = 0.5

    # 阈值设定
    MAX_VECTOR_DISTANCE: float = 0.5  # L2 距离，越小越好
    MIN_CODEBOOK_SCORE: float = 0.5  # 码本分，只要大于0即视为有结构支撑

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

        # 初始化 VQ-VAE
        if vq_model:
            self.vq_model = vq_model
        else:
            self.vq_model = ModelInference(device_str='cuda')

        if not self.vq_model:
            print("⚠️ VQ-VAE 模型加载失败，请检查模型路径。")

        print("智能查询引擎 (Structure Enhanced) 已初始化。")

    # =========================================================================
    # 核心数学辅助函数
    # =========================================================================

    def _calculate_idf(self, code: int, stats: Dict[str, Any]) -> float:
        """
        计算单个 Code 的逆文档频率 (IDF)。
        公式: log( Total_Docs / (Doc_Freq + 1) )
        """
        total_docs = stats.get("total_docs", 0)
        # 如果库是空的，IDF 无意义，返回 0
        if total_docs == 0: return 0.0

        freq_map = stats.get("code_freq", {})
        doc_freq = freq_map.get(code, 0)

        # 使用 +1 平滑防止分母为 0
        idf = math.log(total_docs / (doc_freq + 1))
        return max(0.0, idf)

    def _get_weighted_neighbors(self, source_code: int, stats: Dict[str, Any], top_k: int = 3) -> List[
        Tuple[int, float]]:
        """
        基于共现图谱，获取指定 Code 的加权邻居列表。
        返回: [(neighbor_code, transition_probability), ...]
        """
        graph = stats.get("co_occurrence", {})

        # 如果该 Code 没有邻居记录，返回空
        neighbors_map = graph.get(source_code, {})
        if not neighbors_map: return []

        # 获取源 Code 的总出现次数 (用于计算条件概率 P(B|A) = Count(AB) / Count(A))
        # The denominator uses the source code document frequency.

        source_freq = stats.get("code_freq", {}).get(source_code, 1)

        candidates = []
        for n_code, co_count in neighbors_map.items():
            # 计算转移概率 P(Neighbor | Source)
            prob = co_count / (source_freq + 1)  # +1 防御性编程
            candidates.append((n_code, prob))

        # 按概率降序排序，取 Top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # =========================================================================
    # 策略 2: 初筛 (Retrieval) - 找原子事实
    # =========================================================================
    def _retrieve_atomic_candidates(self, queries: List[str], k_per_step: int, total_atomic_count: int = None) -> List[
        AtomicMemoryUnit]:
        """
        执行多样性检索，返回候选的 AtomicMemoryUnit 列表。
        并在返回前为所有事实计算 vector_score 和 codebook_score。
        """
        final_candidates_map: Dict[str, AtomicMemoryUnit] = {}
        seen_ids: Set[str] = set()

        # [Debug] 获取最新的统计快照 (O(1))
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
            except Exception as e:
                query_embeddings = []

            if future_vq:
                try:
                    query_codes_list = future_vq.result()
                except:
                    query_codes_list = []

        # 取第一条 Query 的结果
        current_embedding = query_embeddings[0] if len(query_embeddings) > 0 else None
        current_codes = query_codes_list[0] if len(query_codes_list) > 0 else []

        # 提前初始化 search_tokens，确保后续普查可用
        search_tokens: Dict[int, float] = {}

        # Path A: Vector Search (语义主路)
        if current_embedding is not None:
            # 这里需要接收 distance (dist)
            vec_results = self.vec_db.search("atomic_memory", current_embedding, k=k_per_step * 2)
            for amu_id, dist in vec_results:
                if len(final_candidates_map) >= 20: break
                if amu_id not in seen_ids:
                    mem = self.meta_db.get_atomic_memory_by_id(amu_id)
                    if mem:
                        # 记录向量分数
                        if mem.metadata is None: mem.metadata = {}
                        mem.metadata['vector_score'] = float(dist)
                        final_candidates_map[amu_id] = mem
                        seen_ids.add(amu_id)

        # Path B: Structure-Enhanced Codebook Search (联想支路)
        # 仅当 VQ 模型启用且有统计数据时执行
        if current_codes and current_stats.get("total_docs", 0) > 0:

            # Step 1: Query 编码加权 (Calculation)
            # 建立搜索令牌集合: {code: weight}
            # (search_tokens 已在上面初始化)

            for q_code in current_codes:
                idf = self._calculate_idf(q_code, current_stats)

                # 过滤低 IDF 的停用词 (如 "is", "the" 对应的 code)
                if idf < self.MIN_IDF_THRESHOLD:
                    continue

                # 基础权重 = IDF
                search_tokens[q_code] = max(search_tokens.get(q_code, 0), idf)

                # Step 2: 图谱联想扩展 (Expansion)
                # 为高权重的词寻找邻居
                neighbors = self._get_weighted_neighbors(q_code, current_stats, top_k=self.EXPANSION_K)

                for n_code, trans_prob in neighbors:
                    # 邻居权重 = 源权重 * 转移概率 * 衰减系数
                    n_weight = idf * trans_prob * self.ALPHA_DECAY

                    # 只有当新权重比已有权重高时才更新 (保留最强路径)
                    if n_weight > search_tokens.get(n_code, 0):
                        search_tokens[n_code] = n_weight

            # Step 3: 倒排索引累积评分 (Map-Reduce)
            candidate_scores: Dict[str, float] = defaultdict(float)

            for code, weight in search_tokens.items():
                # 获取该 Code 下的所有事实 ID
                atomic_ids = self.meta_db.get_atomic_ids_by_code(code)

                for aid in atomic_ids:
                    # 分数累加
                    candidate_scores[aid] += weight

            # Step 4: 长度归一化与排序 (Normalization & Ranking)
            # Rank candidates by the accumulated code-overlap score.
            # This variant does not apply code-length normalization.

            # 按分数降序排列
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

            # Step 5: 补充到最终结果集中
            # 我们用 Path B 来补充 Path A 没找到的 Top-K
            path_b_count = 0
            needed = k_per_step  # Code-path quota.
            needed = 0  # Code-path quota.

            for amu_id, score in sorted_candidates:
                if path_b_count >= needed: break

                if amu_id not in seen_ids:
                    mem = self.meta_db.get_atomic_memory_by_id(amu_id)
                    if mem:
                        # 记录向量分数默认为 999 (因为它是通过码本路径来的，不保证向量距离)
                        if mem.metadata is None: mem.metadata = {}
                        mem.metadata['vector_score'] = 999.0
                        final_candidates_map[amu_id] = mem
                        seen_ids.add(amu_id)
                        path_b_count += 1


        # 全员普查：重新计算所有池内事实的真实码本分
        # 这一步是为了让 Path A (向量) 进来的事实，也能获得 codebook_score
        for amu_id, mem in final_candidates_map.items():
            # 获取该事实的 codes
            fact_codes = getattr(mem, 'code_indices', []) or []

            # 计算该事实与当前 search_tokens 的命中总权重
            recalculated_score = 0.0
            for fc in fact_codes:
                if fc in search_tokens:
                    recalculated_score += search_tokens[fc]

            # 写入元数据
            mem.metadata['codebook_score'] = recalculated_score
            # 确保 vector_score 存在 (防止极端情况)
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
            # 筛选逻辑：检查当前原子事实的分数
            v_score = amu.metadata.get('vector_score', 999.0)
            c_score = amu.metadata.get('codebook_score', 0.0)

            # 只要满足任一条件，即视为有效证据

            is_qualified = v_score < self.MAX_VECTOR_DISTANCE


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
        selected_bases = unique_bases[:25]
        final_atomics = raw_atomics[:40]
        print(f"  - [Selection] Truncated top {len(selected_bases)} 原始对话.")
        print(f"  - [Selection] Truncated top {len(final_atomics)} 原子事实.")

        # 构造 Prompt
        context_str = self._format_evidence(final_atomics, selected_bases)
        final_prompt = self._build_final_prompt(question, context_str, query_date=query_date)

        return {
            "final_prompt": final_prompt,
            "trace_info": trace_info
        }

    def _format_evidence(self, atomics: List[AtomicMemoryUnit], bases: List[BaseMemoryUnit]) -> str:
        # Part 1: Facts
        text = "[EVIDENCE FACTS (Abstracted Insights)]:\n"
        if not atomics: text += "No specific facts extracted.\n"
        for i, f in enumerate(atomics):
            ts_str = ""
            if f.metadata.get('timestamp'):
                ts_str = f"[Message Date: {str(f.metadata.get('timestamp'))[:16]}] "
            text += f"{i + 1}. {f.fact_text}{ts_str}\n"

        # Part 2: Context
        text += "\n[SOURCE CONVERSATION (Original Context)]:\n"
        # 将原始base倒序一下
        bases.sort(key=lambda x: str(x.timestamp) if x.timestamp else "")
        for b in bases:
            speaker = b.metadata.get('speaker', 'Unknown')
            ts = str(b.timestamp)[:16] if b.timestamp else ""
            content = b.text
            text += f"- [{ts}] {speaker}: {content}\n"

        # Instructions
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
