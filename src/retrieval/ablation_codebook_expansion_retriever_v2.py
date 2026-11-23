

import json
import math
import re
import random
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
    """Shuffled direct-code retrieval ablation.

    Path A retains dense-vector retrieval. Path B shuffles the direct-code
    match list, skips already-selected fact IDs and fills its own quota.
    Duplicate occurrences in the list are retained before shuffling.
    IDF weighting and graph expansion are disabled; selected code-path
    facts receive the constant score 0.35. Randomness is not seeded here."""

    ALPHA_DECAY: float = 0.3
    EXPANSION_K: int = 3
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

        print("智能查询引擎 (Ablation: Stochastic Collision Mode) 已初始化。")

    def _calculate_idf(self, code: int, stats: Dict[str, Any]) -> float:
        return 1.0  # 强制失效

    def _get_weighted_neighbors(self, source_code: int, stats: Dict[str, Any], top_k: int = 3) -> List[
        Tuple[int, float]]:
        return []  # 彻底切断扩展链条

    def _retrieve_atomic_candidates(self, queries: List[str], k_per_step: int, total_atomic_count: int = None) -> List[
        AtomicMemoryUnit]:
        """Retrieve dense candidates and shuffled direct-code matches.

        The code-path list is shuffled without relevance ranking, then selected
        up to its quota with fact-ID deduplication. Code-path scores are fixed
        at 0.35; this is a sampling ablation, not the full codebook ranker."""
        final_candidates_map: Dict[str, AtomicMemoryUnit] = {}
        seen_ids: Set[str] = set()
        current_stats = self.meta_db.get_stats()

        # 模型推理
        query_embeddings = []
        query_codes_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_embed = executor.submit(self.embedder.embed, queries, is_query=True)
            if self.vq_model:
                future_vq = executor.submit(self.vq_model.get_indices, queries)

            try:
                query_embeddings = future_embed.result()
            except:
                query_embeddings = []
            if self.vq_model:
                try:
                    query_codes_list = future_vq.result()
                except:
                    query_codes_list = []

        current_embedding = query_embeddings[0] if len(query_embeddings) > 0 else None
        current_codes = query_codes_list[0] if len(query_codes_list) > 0 else []

        # Path A: Vector Search (语义主路保持作为 Baseline 对照)
        if current_embedding is not None:
            vec_results = self.vec_db.search("atomic_memory", current_embedding, k=k_per_step * 2)
            for amu_id, dist in vec_results:
                if len(final_candidates_map) >= k_per_step: break
                mem = self.meta_db.get_atomic_memory_by_id(amu_id)
                if mem:
                    if mem.metadata is None: mem.metadata = {}
                    mem.metadata['vector_score'] = float(dist)
                    final_candidates_map[amu_id] = mem
                    seen_ids.add(amu_id)

        # Path B: shuffled direct-code matches without IDF or graph expansion.
        if current_codes and current_stats.get("total_docs", 0) > 0:

            candidate_pool = []
            # Collect direct code matches without IDF weighting.
            for q_code in current_codes:
                # 获取该 Code 对应的所有事实
                atomic_ids = self.meta_db.get_atomic_ids_by_code(q_code)
                candidate_pool.extend(atomic_ids)

            # Shuffle the candidate list before quota-based deduplicated selection.
            # The list retains duplicate occurrences from multiple matching codes.
            # No graph expansion or relevance ordering is applied to the code-path list.
            random.shuffle(candidate_pool)

            path_b_count = 0
            for amu_id in candidate_pool:
                if path_b_count >= k_per_step: break
                if amu_id not in seen_ids:
                    mem = self.meta_db.get_atomic_memory_by_id(amu_id)
                    if mem:
                        if mem.metadata is None: mem.metadata = {}
                        mem.metadata['vector_score'] = 999.0
                        # Assign every code-path selection the same score (0.35).
                        mem.metadata['codebook_score'] = 0.35
                        final_candidates_map[amu_id] = mem
                        seen_ids.add(amu_id)
                        path_b_count += 1

        # 全员分数补齐
        for amu_id, mem in final_candidates_map.items():
            if 'codebook_score' not in mem.metadata:
                mem.metadata['codebook_score'] = 0.0
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
                    if base_mem: base_map[base_id] = base_mem

        ordered_unique_bases = []
        added_base_ids = set()

        for amu in atomics:
            v_score = amu.metadata.get('vector_score', 999.0)
            c_score = amu.metadata.get('codebook_score', 0.0)

            # Apply the stored vector-distance or code-score evidence threshold.
            is_qualified = (v_score < self.MAX_VECTOR_DISTANCE) or (c_score > self.MIN_CODEBOOK_SCORE)

            if is_qualified:
                for base_id in amu.source_base_ids:
                    if base_id in base_map and base_id not in added_base_ids:
                        ordered_unique_bases.append(base_map[base_id])
                        added_base_ids.add(base_id)

        return ordered_unique_bases, base_to_atomics_map

    def query(self, question: str, k: int = 5, total_atomic_count: int = None, query_date: str = None, **kwargs) -> \
    Dict[str, Any]:
        print(f"\n--- Querying (Ablation: Stochastic Collision Mode): '{question}' ---")

        trace_info = {
            "question": question,
            "strategies_used": ["direct_retrieval", "stochastic_code_collision"],
            "expanded_queries": [question],
            "retrieved_pool_size": 0
        }

        k_initial = 20
        raw_atomics = self._retrieve_atomic_candidates([question], k_per_step=k_initial,
                                                       total_atomic_count=total_atomic_count)
        trace_info["retrieved_pool_size"] = len(raw_atomics)

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
