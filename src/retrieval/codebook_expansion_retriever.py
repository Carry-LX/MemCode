import json
import math
import re
from typing import List, Any, Dict, Set, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import concurrent.futures
import torch
import torch.nn.functional as F

from src.utils.vqvae_inference_v10 import ModelInference
from src.core.memory_structures import BaseMemoryUnit, AtomicMemoryUnit
from src.core.metadata_db_handler import MetadataDBHandler
from src.core.vector_db_handler import VectorDBHandler
from src.utils.embedding_bge import EmbeddingModel


class QueryEngine:
    """
    智能查询引擎 - 语义拓扑增强 & 归一化重排版 (Semantic Topology & Re-ranking)

    核心机制:
    1. [语义层] 利用预计算的 Codebook 相似度矩阵，进行向量空间的软匹配 (Soft-Matching)，解决 VQ 量化噪声。
    2. [结构层] 基于共现图谱进行联想扩展 (Graph Expansion)。
    3. [排序层] 对码本检索结果执行 "粗排 -> 长度归一化 -> 精排"，解决长文本得分偏见问题。
    """

    # [Hyperparameters] 超参数定义

    # 1. 图谱扩展参数 (Graph Expansion)
    # 衰减系数: 控制图谱扩展邻居的权重
    ALPHA_DECAY: float = 0.3
    # 扩展数量: 每个 Code 在共现图谱中最多扩展多少个强关联邻居
    EXPANSION_K: int = 3

    # 2. 语义扩展参数 (Semantic Expansion)
    # 语义相似度阈值: 只有 Cosine Similarity 高于此值的 Code 才会被视为语义邻居。
    # The cosine similarity cutoff is configured below.
    SEMANTIC_SIM_THRESHOLD: float = 0.75

    # 语义扩展数量: 每个 Code 在向量空间中最多找多少个近邻。
    SEMANTIC_EXPANSION_K: int = 3

    # 语义权重衰减: 语义邻居的权重系数。
    # Scale semantic-neighbor contributions by this decay factor.
    BETA_SEMANTIC_DECAY: float = 0.9

    # 3. 基础过滤参数
    # 最小 IDF 阈值: 忽略 IDF 低于此值的 Code (过滤停用语义)
    MIN_IDF_THRESHOLD: float = 0.5

    # 4. 重排序参数 (Re-ranking)
    # 粗排数量: 从倒排索引累分结果中取多少个高分候选用作归一化精排
    RERANK_POOL_SIZE: int = 100

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

        # [Cache] 语义相似度矩阵缓存
        self.code_sim_matrix = None

        # 初始化 VQ-VAE
        if vq_model:
            self.vq_model = vq_model
        else:
            self.vq_model = ModelInference(device_str='cuda')

        if not self.vq_model:
            print("⚠️ VQ-VAE 模型加载失败，请检查模型路径。")
        else:
            # 预计算 Codebook 相似度矩阵
            self._precompute_codebook_similarity()

        print("智能查询引擎 (Structure & Semantic Enhanced) 已初始化。")

    def _precompute_codebook_similarity(self):
        """
        [Init] 从 VQ-VAE 模型中提取 Codebook 权重，并计算全量相似度矩阵。
        """
        try:
            # 路径依据: vq_model.model (VQVAE_T5) -> vq_layer (ImprovedVectorQuantizer)
            # > codebook (EuclideanCodebook) -> embedding (nn.Embedding)
            codebook_weight = self.vq_model.model.vq_layer.codebook.embedding.weight.data  # [1024, Dim]

            # 归一化以准备计算 Cosine Similarity
            norm_weight = F.normalize(codebook_weight, p=2, dim=-1)

            # 计算 Gram Matrix: [1024, 1024]
            # sim_matrix[i, j] 即 Code i 和 Code j 的余弦相似度
            self.code_sim_matrix = torch.mm(norm_weight, norm_weight.t())

            # 将对角线 (自己与自己) 设为 -1.0，防止检索时把自己作为邻居返回
            self.code_sim_matrix.fill_diagonal_(-1.0)

            print(f"✅ [QueryEngine] Codebook 语义拓扑矩阵已构建。Shape: {self.code_sim_matrix.shape}")

        except AttributeError as e:
            print(f"⚠️ [QueryEngine] 无法访问 Codebook 权重，语义扩展功能将不可用。Err: {e}")
            self.code_sim_matrix = None
        except Exception as e:
            print(f"⚠️ [QueryEngine] 相似度矩阵计算失败。Err: {e}")
            self.code_sim_matrix = None

    # =========================================================================
    # [Helper] 语义检索辅助函数
    # =========================================================================

    def _get_semantic_neighbors(self, source_code: int, top_k: int = 2) -> List[Tuple[int, float]]:
        """
        基于向量空间相似度，获取指定 Code 的语义邻居。
        返回: [(neighbor_code, similarity_score), ...]
        """
        if self.code_sim_matrix is None:
            return []

        # 边界检查
        if source_code < 0 or source_code >= self.code_sim_matrix.size(0):
            return []

        # 获取该行相似度
        sim_scores = self.code_sim_matrix[source_code]  # [1024]

        # 筛选大于阈值的候选者
        mask = sim_scores > self.SEMANTIC_SIM_THRESHOLD
        valid_indices = torch.nonzero(mask).squeeze(1)

        if valid_indices.numel() == 0:
            return []

        valid_scores = sim_scores[valid_indices]

        # 排序并取 Top-K (descending)
        sorted_scores, sorted_idx_indices = torch.sort(valid_scores, descending=True)

        top_k = min(top_k, len(sorted_scores))

        results = []
        for i in range(top_k):
            score = sorted_scores[i].item()
            real_idx = valid_indices[sorted_idx_indices[i]].item()
            results.append((real_idx, score))

        return results

    # =========================================================================
    # [Helper] 数学统计辅助函数
    # =========================================================================

    def _calculate_idf(self, code: int, stats: Dict[str, Any]) -> float:
        """
        计算单个 Code 的逆文档频率 (IDF)。
        公式: log( Total_Docs / (Doc_Freq + 1) )
        """
        total_docs = stats.get("total_docs", 0)
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
        neighbors_map = graph.get(source_code, {})
        if not neighbors_map: return []

        source_freq = stats.get("code_freq", {}).get(source_code, 1)

        candidates = []
        for n_code, co_count in neighbors_map.items():
            # 计算转移概率 P(Neighbor | Source)
            prob = co_count / (source_freq + 1)
            candidates.append((n_code, prob))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # =========================================================================
    # 策略 2: 初筛 (Retrieval) - 找原子事实
    # =========================================================================
    def _retrieve_atomic_candidates(self, queries: List[str], k_per_step: int, total_atomic_count: int = None) -> List[
        AtomicMemoryUnit]:
        """
        执行多样性检索，返回候选的 AtomicMemoryUnit 列表。
        Path A: 向量检索 (标准)
        Path B: 语义增强码本检索 (级联扩展 + 归一化重排)
        """
        final_candidates_map: Dict[str, AtomicMemoryUnit] = {}
        seen_ids: Set[str] = set()

        # [Debug] 获取最新的统计快照
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

        # Path A: Vector Search (语义主路)
        if current_embedding is not None:
            vec_results = self.vec_db.search("atomic_memory", current_embedding, k=k_per_step * 2)
            for amu_id, _ in vec_results:
                if len(final_candidates_map) >= k_per_step: break
                if amu_id not in seen_ids:
                    mem = self.meta_db.get_atomic_memory_by_id(amu_id)
                    if mem:
                        final_candidates_map[amu_id] = mem
                        seen_ids.add(amu_id)

        # Path B: Semantic & Graph Enhanced Codebook Search (联想支路)
        if current_codes and current_stats.get("total_docs", 0) > 0:

            search_tokens: Dict[int, float] = {}

            # Stage 0 & 1: 原始 Code 权重计算 & 语义软匹配 (Semantic Soft-Matching)
            for q_code in current_codes:
                idf = self._calculate_idf(q_code, current_stats)

                # 过滤低 IDF 的停用词
                if idf < self.MIN_IDF_THRESHOLD:
                    continue

                # 1. 记录原始词权重
                current_w = max(search_tokens.get(q_code, 0), idf)
                search_tokens[q_code] = current_w

                # 2. 语义扩展 (Vector Space Expansion)
                # 只有当相似度矩阵存在时才执行
                if self.code_sim_matrix is not None:
                    sem_neighbors = self._get_semantic_neighbors(q_code, top_k=self.SEMANTIC_EXPANSION_K)

                    for n_code, similarity in sem_neighbors:
                        # 语义邻居权重 = 父节点权重 * 相似度 * 语义衰减系数
                        sem_weight = idf * similarity * self.BETA_SEMANTIC_DECAY

                        if sem_weight > search_tokens.get(n_code, 0):
                            search_tokens[n_code] = sem_weight

            # Stage 2: 图谱联想扩展 (Graph Expansion)
            # 对当前池中所有的 Code (包括语义近邻) 进行共现图扩展
            current_tokens_list = list(search_tokens.items())

            for token_code, token_weight in current_tokens_list:
                neighbors = self._get_weighted_neighbors(token_code, current_stats, top_k=self.EXPANSION_K)

                for n_code, trans_prob in neighbors:
                    # 邻居权重 = 源权重 * 转移概率 * 图谱衰减系数
                    n_weight = token_weight * trans_prob * self.ALPHA_DECAY

                    if n_weight > search_tokens.get(n_code, 0):
                        search_tokens[n_code] = n_weight

            # Step 3: 倒排索引累积评分 (Map-Reduce)
            candidate_scores: Dict[str, float] = defaultdict(float)

            for code, weight in search_tokens.items():
                atomic_ids = self.meta_db.get_atomic_ids_by_code(code)
                for aid in atomic_ids:
                    candidate_scores[aid] += weight

            # Step 4: 粗排 (Coarse Ranking)
            # 仅取 Top-N 个高分候选进行昂贵的归一化计算
            sorted_raw = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            candidates_pool = sorted_raw[:self.RERANK_POOL_SIZE]

            # Step 5: 归一化与精排 (Normalization & Re-ranking)
            reranked_candidates = []

            for amu_id, raw_score in candidates_pool:
                # 获取内存对象以计算长度
                # 优化: 如果已经在 seen_ids 中 (Path A 检索过)，直接复用
                if amu_id in final_candidates_map:
                    mem = final_candidates_map[amu_id]
                else:
                    mem = self.meta_db.get_atomic_memory_by_id(amu_id)

                if not mem: continue

                # 获取 Code 长度
                code_len = len(mem.code_indices) if mem.code_indices else 1
                if code_len < 1: code_len = 1

                # 归一化公式: Score / sqrt(Length)
                # 惩罚长文本，奖励短而精的匹配
                norm_score = raw_score / (math.sqrt(code_len) + 1e-6)

                reranked_candidates.append((amu_id, mem, norm_score))

            # 按归一化分数精排
            reranked_candidates.sort(key=lambda x: x[2], reverse=True)

            # Step 6: 补充到最终结果集 (Supplement)
            path_b_count = 0
            needed = k_per_step

            for amu_id, mem, score in reranked_candidates:
                if path_b_count >= needed: break

                if amu_id not in seen_ids:
                    final_candidates_map[amu_id] = mem
                    seen_ids.add(amu_id)
                    path_b_count += 1

            # [Optional Debug]


        return list(final_candidates_map.values())

    # =========================================================================
    # 核心转换逻辑: Atomics -> Unique Bases
    # =========================================================================
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
            "strategies_used": ["direct_retrieval", "semantic_topology_expansion", "graph_expansion",
                                "reranking_normalization"],
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
