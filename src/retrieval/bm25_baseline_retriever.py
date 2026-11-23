import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

from src.core.memory_structures import AtomicMemoryUnit, BaseMemoryUnit
from src.core.metadata_db_handler import MetadataDBHandler
from src.core.vector_db_handler import VectorDBHandler
from src.utils.embedding_bge import EmbeddingModel


class QueryEngine:
    """
    BM25-augmented retrieval baseline.

    Path A: dense vector retrieval
    Path B: untuned BM25 retrieval over atomic facts
    """

    MAX_VECTOR_DISTANCE: float = 0.65
    MIN_BM25_SCORE: float = 0.25
    VECTOR_TRACE_POOL_K: int = 100
    BM25_FILL_K: int = 20
    BM25_K1: float = 1.2
    BM25_B: float = 0.75

    STOPWORDS: Set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "did", "do", "does", "for", "from", "had", "has", "have",
        "how", "i", "in", "into", "is", "it", "its", "me", "my", "of", "on",
        "or", "our", "she", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "those", "to", "was", "were", "what", "when",
        "where", "which", "who", "why", "with", "would", "you", "your",
        "during", "recently", "mention", "mentioned", "kind", "type", "didn",
        "doesn", "about", "besides", "after", "before", "into", "while",
    }

    def __init__(
        self,
        embedder: EmbeddingModel,
        meta_db: MetadataDBHandler,
        vec_db: VectorDBHandler,
        llm_interface: Any,
        vq_model: Any = None,
        stop_ratio: float = 0.15,
    ):
        self.embedder = embedder
        self.meta_db = meta_db
        self.vec_db = vec_db
        self.llm = llm_interface
        self.stop_ratio = stop_ratio
        self.vq_model = vq_model

        self._bm25_cache_atomic_count = -1
        self._bm25_df: Counter = Counter()
        self._bm25_atomic_term_freq: Dict[str, Counter] = {}
        self._bm25_inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self._bm25_doc_length: Dict[str, int] = {}
        self._bm25_atomic_count = 0
        self._bm25_avg_doc_length = 0.0

        print("BM25 retrieval baseline initialized.")

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [tok for tok in tokens if len(tok) >= 2 and tok not in self.STOPWORDS]

    def _ensure_bm25_cache(self) -> None:
        current_atomic_count = self.meta_db.get_total_atomic_count()
        if current_atomic_count == self._bm25_cache_atomic_count:
            return

        atomics = self.meta_db.get_all_active_atomics()
        bm25_df: Counter = Counter()
        atomic_term_freq: Dict[str, Counter] = {}
        inverted_index: Dict[str, Set[str]] = defaultdict(set)
        doc_length_map: Dict[str, int] = {}
        total_doc_length = 0

        for amu in atomics:
            tokens = self._tokenize(amu.fact_text)
            if not tokens:
                continue
            term_freq = Counter(tokens)
            atomic_term_freq[amu.id] = term_freq
            doc_length_map[amu.id] = len(tokens)
            total_doc_length += len(tokens)
            for token in term_freq.keys():
                bm25_df[token] += 1
                inverted_index[token].add(amu.id)

        self._bm25_df = bm25_df
        self._bm25_atomic_term_freq = atomic_term_freq
        self._bm25_inverted_index = inverted_index
        self._bm25_doc_length = doc_length_map
        self._bm25_atomic_count = len(atomics)
        self._bm25_avg_doc_length = (total_doc_length / len(doc_length_map)) if doc_length_map else 0.0
        self._bm25_cache_atomic_count = current_atomic_count

    def _bm25_idf(self, token: str) -> float:
        if self._bm25_atomic_count == 0:
            return 0.0
        df = self._bm25_df.get(token, 0)
        numerator = self._bm25_atomic_count - df + 0.5
        denominator = df + 0.5
        return math.log(1.0 + (numerator / denominator))

    def _retrieve_atomic_candidates(
        self,
        queries: List[str],
        k_per_step: int,
        total_atomic_count: int = None,
    ) -> Tuple[List[AtomicMemoryUnit], Dict[str, Any]]:
        final_candidates_map: Dict[str, AtomicMemoryUnit] = {}
        seen_ids: Set[str] = set()

        retrieval_debug: Dict[str, Any] = {
            "bm25_terms": [],
            "vector_stage_k": k_per_step,
            "vector_trace_pool_k": max(k_per_step * 2, self.VECTOR_TRACE_POOL_K),
            "vector_trace_pool_size": 0,
            "vector_stage_selected_count": 0,
            "bm25_candidate_pool_size": 0,
            "bm25_injected_count": 0,
            "bm25_term_count": 0,
            "bm25_terms_top": [],
            "bm25_k1": self.BM25_K1,
            "bm25_b": self.BM25_B,
            "bm25_avg_doc_length": 0.0,
        }

        current_query = queries[0] if queries else ""
        current_embedding = None
        if queries:
            try:
                embedded = self.embedder.embed(queries, is_query=True)
                current_embedding = embedded[0] if embedded else None
            except Exception:
                current_embedding = None

        query_terms = self._tokenize(current_query)
        retrieval_debug["bm25_terms"] = query_terms
        retrieval_debug["bm25_term_count"] = len(query_terms)

        vector_raw_rank_map: Dict[str, int] = {}
        vector_raw_dist_map: Dict[str, float] = {}
        vector_stage_ids: List[str] = []
        bm25_raw_rank_map: Dict[str, int] = {}
        bm25_score_map: Dict[str, float] = {}
        bm25_hit_tokens_map: Dict[str, List[str]] = {}
        bm25_injected_ids: List[str] = []

        if current_embedding is not None:
            trace_pool_k = retrieval_debug["vector_trace_pool_k"]
            vec_results = self.vec_db.search("atomic_memory", current_embedding, k=trace_pool_k)
            retrieval_debug["vector_trace_pool_size"] = len(vec_results)

            for rank, (amu_id, dist) in enumerate(vec_results, start=1):
                vector_raw_rank_map[amu_id] = rank
                vector_raw_dist_map[amu_id] = float(dist)

            for rank, (amu_id, dist) in enumerate(vec_results, start=1):
                if len(final_candidates_map) >= k_per_step:
                    break
                if amu_id in seen_ids:
                    continue
                mem = self.meta_db.get_atomic_memory_by_id(amu_id)
                if not mem:
                    continue
                if mem.metadata is None:
                    mem.metadata = {}
                mem.metadata["vector_score"] = float(dist)
                mem.metadata["selected_by_vector_stage"] = True
                mem.metadata["vector_stage_rank"] = rank
                final_candidates_map[amu_id] = mem
                seen_ids.add(amu_id)
                vector_stage_ids.append(amu_id)

        self._ensure_bm25_cache()
        retrieval_debug["bm25_avg_doc_length"] = float(self._bm25_avg_doc_length)
        candidate_bm25_scores: Dict[str, float] = defaultdict(float)
        candidate_bm25_hits: Dict[str, Set[str]] = defaultdict(set)

        term_rank_view: List[Tuple[str, float]] = []
        seen_terms: Set[str] = set()
        for token in query_terms:
            if token in seen_terms:
                continue
            seen_terms.add(token)
            idf = self._bm25_idf(token)
            term_rank_view.append((token, idf))
            atomic_ids = self._bm25_inverted_index.get(token, set())
            for amu_id in atomic_ids:
                tf_counter = self._bm25_atomic_term_freq.get(amu_id)
                if not tf_counter:
                    continue
                tf = tf_counter.get(token, 0)
                if tf <= 0:
                    continue
                doc_length = self._bm25_doc_length.get(amu_id, 0)
                if doc_length <= 0 or self._bm25_avg_doc_length <= 0:
                    continue
                norm = self.BM25_K1 * (1.0 - self.BM25_B + self.BM25_B * (doc_length / self._bm25_avg_doc_length))
                token_score = idf * ((tf * (self.BM25_K1 + 1.0)) / (tf + norm))
                candidate_bm25_scores[amu_id] += token_score
                candidate_bm25_hits[amu_id].add(token)

        retrieval_debug["bm25_terms_top"] = [
            {"token": token, "weight": float(score)}
            for token, score in sorted(term_rank_view, key=lambda x: x[1], reverse=True)[:20]
        ]

        sorted_bm25_candidates = sorted(candidate_bm25_scores.items(), key=lambda x: x[1], reverse=True)
        retrieval_debug["bm25_candidate_pool_size"] = len(sorted_bm25_candidates)

        for rank, (amu_id, score) in enumerate(sorted_bm25_candidates, start=1):
            bm25_raw_rank_map[amu_id] = rank
            bm25_score_map[amu_id] = float(score)
            bm25_hit_tokens_map[amu_id] = sorted(candidate_bm25_hits.get(amu_id, set()))

        bm25_fill_count = 0
        for amu_id, score in sorted_bm25_candidates:
            if bm25_fill_count >= self.BM25_FILL_K:
                break
            if score < self.MIN_BM25_SCORE:
                continue
            if amu_id in seen_ids:
                continue
            mem = self.meta_db.get_atomic_memory_by_id(amu_id)
            if not mem:
                continue
            if mem.metadata is None:
                mem.metadata = {}
            mem.metadata["vector_score"] = 999.0
            mem.metadata["injected_by_bm25_stage"] = True
            mem.metadata["bm25_injection_rank"] = bm25_fill_count + 1
            final_candidates_map[amu_id] = mem
            seen_ids.add(amu_id)
            bm25_injected_ids.append(amu_id)
            bm25_fill_count += 1

        for amu_id, mem in final_candidates_map.items():
            bm25_score = float(bm25_score_map.get(amu_id, 0.0))
            hit_tokens = bm25_hit_tokens_map.get(amu_id, [])

            mem.metadata["bm25_score"] = bm25_score
            mem.metadata["bm25_hit_token_count"] = len(hit_tokens)
            mem.metadata["bm25_hit_tokens"] = hit_tokens
            mem.metadata["selected_by_vector_stage"] = amu_id in vector_stage_ids
            mem.metadata["injected_by_bm25_stage"] = amu_id in bm25_injected_ids
            mem.metadata["vector_raw_rank_at_100"] = vector_raw_rank_map.get(amu_id)
            mem.metadata["vector_raw_dist_at_100"] = vector_raw_dist_map.get(amu_id)
            mem.metadata["bm25_raw_rank"] = bm25_raw_rank_map.get(amu_id)
            if "vector_score" not in mem.metadata:
                mem.metadata["vector_score"] = 999.0

        retrieval_debug["vector_stage_selected_count"] = len(vector_stage_ids)
        retrieval_debug["bm25_injected_count"] = len(bm25_injected_ids)
        return list(final_candidates_map.values()), retrieval_debug

    def _group_and_resolve_to_base(
        self, atomics: List[AtomicMemoryUnit]
    ) -> Tuple[List[BaseMemoryUnit], Dict[str, List[AtomicMemoryUnit]]]:
        base_map: Dict[str, BaseMemoryUnit] = {}
        base_to_atomics_map = defaultdict(list)

        for amu in atomics:
            if not amu.source_base_ids:
                continue
            for base_id in amu.source_base_ids:
                base_to_atomics_map[base_id].append(amu)
                if base_id not in base_map:
                    base_mem = self.meta_db.get_base_memory_by_id(base_id)
                    if base_mem:
                        base_map[base_id] = base_mem

        ordered_unique_bases = []
        added_base_ids = set()

        for amu in atomics:
            v_score = amu.metadata.get("vector_score", 999.0)
            b_score = amu.metadata.get("bm25_score", 0.0)
            is_qualified = (v_score < self.MAX_VECTOR_DISTANCE) or (b_score > self.MIN_BM25_SCORE)
            if is_qualified:
                for base_id in amu.source_base_ids:
                    if base_id in base_map and base_id not in added_base_ids:
                        ordered_unique_bases.append(base_map[base_id])
                        added_base_ids.add(base_id)

        return ordered_unique_bases, base_to_atomics_map

    def _normalize_time_bucket(self, timestamp_value: Any) -> str:
        if not timestamp_value:
            return ""
        if isinstance(timestamp_value, datetime):
            return timestamp_value.isoformat()[:10]
        ts_str = str(timestamp_value).strip()
        return ts_str[:10] if len(ts_str) >= 10 else ts_str

    def _get_base_memory_cached(self, base_id: str, cache: Dict[str, BaseMemoryUnit]) -> Any:
        if base_id not in cache:
            cache[base_id] = self.meta_db.get_base_memory_by_id(base_id)
        return cache[base_id]

    def _build_candidate_support_trace(
        self, atomics: List[AtomicMemoryUnit]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        candidate_trace: List[Dict[str, Any]] = []
        base_cache: Dict[str, BaseMemoryUnit] = {}

        vector_stage_base_ids: Set[str] = set()
        vector_stage_time_buckets: Set[str] = set()
        for amu in atomics:
            if amu.metadata.get("selected_by_vector_stage", False):
                for base_id in amu.source_base_ids or []:
                    vector_stage_base_ids.add(base_id)
                    base_mem = self._get_base_memory_cached(base_id, base_cache)
                    if base_mem and base_mem.timestamp:
                        bucket = self._normalize_time_bucket(base_mem.timestamp)
                        if bucket:
                            vector_stage_time_buckets.add(bucket)

        final_base_ids: Set[str] = set()
        final_time_buckets: Set[str] = set()
        support_type_counter: Counter = Counter()
        origin_type_counter: Counter = Counter()

        bm25_injected_late_vector_hit_count = 0
        bm25_injected_true_vector_miss_count = 0
        bm25_injected_new_base_count = 0
        bm25_injected_new_time_count = 0
        bm25_injected_redundant_count = 0

        for rank, amu in enumerate(atomics, start=1):
            v_score = float(amu.metadata.get("vector_score", 999.0))
            b_score = float(amu.metadata.get("bm25_score", 0.0))
            vector_supported = v_score < self.MAX_VECTOR_DISTANCE
            bm25_supported = b_score > self.MIN_BM25_SCORE

            if vector_supported and bm25_supported:
                support_type = "both_supported"
            elif vector_supported:
                support_type = "vector_only_supported"
            elif bm25_supported:
                support_type = "bm25_only_supported"
            else:
                support_type = "weak_supported"

            selected_by_vector_stage = bool(amu.metadata.get("selected_by_vector_stage", False))
            injected_by_bm25_stage = bool(amu.metadata.get("injected_by_bm25_stage", False))
            if injected_by_bm25_stage:
                origin_type = "bm25_injected"
            elif selected_by_vector_stage:
                origin_type = "vector_seeded"
            else:
                origin_type = "unknown_origin"

            vector_raw_rank = amu.metadata.get("vector_raw_rank_at_100")
            if selected_by_vector_stage:
                vector_rank_bucket = "vector_topk_selected"
            elif vector_raw_rank is not None:
                vector_rank_bucket = "late_vector_hit_within_top100"
            else:
                vector_rank_bucket = "true_vector_miss_beyond_top100"

            source_base_ids = list(amu.source_base_ids or [])
            source_time_buckets: List[str] = []
            novel_base_ids: List[str] = []
            novel_time_buckets: List[str] = []

            for base_id in source_base_ids:
                final_base_ids.add(base_id)
                base_mem = self._get_base_memory_cached(base_id, base_cache)
                if base_id not in vector_stage_base_ids:
                    novel_base_ids.append(base_id)
                if base_mem and base_mem.timestamp:
                    bucket = self._normalize_time_bucket(base_mem.timestamp)
                    if bucket:
                        source_time_buckets.append(bucket)
                        final_time_buckets.add(bucket)
                        if bucket not in vector_stage_time_buckets:
                            novel_time_buckets.append(bucket)

            source_time_buckets = sorted(set(source_time_buckets))
            novel_base_ids = sorted(set(novel_base_ids))
            novel_time_buckets = sorted(set(novel_time_buckets))

            if injected_by_bm25_stage:
                if vector_raw_rank is not None:
                    bm25_injected_late_vector_hit_count += 1
                else:
                    bm25_injected_true_vector_miss_count += 1

                if novel_base_ids:
                    bm25_injected_new_base_count += 1
                if novel_time_buckets:
                    bm25_injected_new_time_count += 1
                if (not novel_base_ids) and (not novel_time_buckets) and vector_raw_rank is not None:
                    bm25_injected_redundant_count += 1

            support_type_counter[support_type] += 1
            origin_type_counter[origin_type] += 1

            candidate_trace.append({
                "rank": rank,
                "atomic_id": amu.id,
                "fact_text": amu.fact_text,
                "fact_text_short": amu.fact_text[:180],
                "vector_score": v_score,
                "bm25_score": b_score,
                "vector_supported": vector_supported,
                "bm25_supported": bm25_supported,
                "support_type": support_type,
                "origin_type": origin_type,
                "selected_by_vector_stage": selected_by_vector_stage,
                "injected_by_bm25_stage": injected_by_bm25_stage,
                "vector_stage_rank": amu.metadata.get("vector_stage_rank"),
                "vector_raw_rank_at_100": vector_raw_rank,
                "vector_raw_dist_at_100": amu.metadata.get("vector_raw_dist_at_100"),
                "vector_rank_bucket": vector_rank_bucket,
                "bm25_raw_rank": amu.metadata.get("bm25_raw_rank"),
                "bm25_hit_token_count": int(amu.metadata.get("bm25_hit_token_count", 0)),
                "bm25_hit_tokens": list(amu.metadata.get("bm25_hit_tokens", [])),
                "source_base_ids": source_base_ids,
                "source_time_buckets": source_time_buckets,
                "novel_base_ids_vs_vector_stage": novel_base_ids,
                "has_novel_base_vs_vector_stage": len(novel_base_ids) > 0,
                "novel_time_buckets_vs_vector_stage": novel_time_buckets,
                "has_novel_time_vs_vector_stage": len(novel_time_buckets) > 0,
                "code_count": len(getattr(amu, "code_indices", []) or []),
            })

        support_type_counts = {
            "both_supported": int(support_type_counter.get("both_supported", 0)),
            "vector_only_supported": int(support_type_counter.get("vector_only_supported", 0)),
            "bm25_only_supported": int(support_type_counter.get("bm25_only_supported", 0)),
            "weak_supported": int(support_type_counter.get("weak_supported", 0)),
        }
        origin_type_counts = {
            "vector_seeded": int(origin_type_counter.get("vector_seeded", 0)),
            "bm25_injected": int(origin_type_counter.get("bm25_injected", 0)),
            "unknown_origin": int(origin_type_counter.get("unknown_origin", 0)),
        }

        summary: Dict[str, Any] = {
            "both_supported": support_type_counts["both_supported"],
            "vector_only_supported": support_type_counts["vector_only_supported"],
            "bm25_only_supported": support_type_counts["bm25_only_supported"],
            "weak_supported": support_type_counts["weak_supported"],
            "vector_seeded": origin_type_counts["vector_seeded"],
            "bm25_injected": origin_type_counts["bm25_injected"],
            "unknown_origin": origin_type_counts["unknown_origin"],
            "support_type_counts": support_type_counts,
            "origin_type_counts": origin_type_counts,
            "vector_stage_base_count": len(vector_stage_base_ids),
            "final_base_count": len(final_base_ids),
            "bm25_added_new_base_count": len(final_base_ids - vector_stage_base_ids),
            "vector_stage_time_bucket_count": len(vector_stage_time_buckets),
            "final_time_bucket_count": len(final_time_buckets),
            "bm25_added_new_time_bucket_count": len(final_time_buckets - vector_stage_time_buckets),
            "bm25_injected_late_vector_hit_count": bm25_injected_late_vector_hit_count,
            "bm25_injected_true_vector_miss_count": bm25_injected_true_vector_miss_count,
            "bm25_injected_new_base_count": bm25_injected_new_base_count,
            "bm25_injected_new_time_count": bm25_injected_new_time_count,
            "bm25_injected_redundant_count": bm25_injected_redundant_count,
        }
        return candidate_trace, summary

    def query(
        self,
        question: str,
        k: int = 5,
        total_atomic_count: int = None,
        query_date: str = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        print(f"\n--- Querying (BM25 Baseline): '{question}' ---")

        trace_info = {
            "question": question,
            "strategies_used": ["direct_retrieval", "bm25_inverted_fill"],
            "expanded_queries": [],
            "retrieved_pool_size": 0,
        }

        queries = [question]
        trace_info["expanded_queries"] = queries

        k = 20
        raw_atomics, retrieval_debug = self._retrieve_atomic_candidates(
            queries,
            k_per_step=k,
            total_atomic_count=total_atomic_count,
        )
        trace_info["retrieved_pool_size"] = len(raw_atomics)
        candidate_trace, candidate_summary = self._build_candidate_support_trace(raw_atomics)
        trace_info["candidate_support"] = candidate_trace
        trace_info["candidate_support_summary"] = candidate_summary
        trace_info["question_level_tags"] = candidate_summary
        trace_info["retrieval_stage_trace"] = retrieval_debug
        trace_info["candidate_support_thresholds"] = {
            "max_vector_distance": self.MAX_VECTOR_DISTANCE,
            "min_bm25_score": self.MIN_BM25_SCORE,
            "vector_trace_pool_k": self.VECTOR_TRACE_POOL_K,
        }
        print(f"  - Retrieved atomic facts: {len(raw_atomics)}")

        unique_bases, _ = self._group_and_resolve_to_base(raw_atomics)
        selected_bases = unique_bases[:25]
        final_atomics = raw_atomics[:40]
        print(f"  - [Selection] Truncated top {len(selected_bases)} base memories.")
        print(f"  - [Selection] Truncated top {len(final_atomics)} atomic facts.")

        context_str = self._format_evidence(final_atomics, selected_bases)
        final_prompt = self._build_final_prompt(question, context_str, query_date=query_date)
        return {
            "final_prompt": final_prompt,
            "trace_info": trace_info,
        }

    def _format_evidence(self, atomics: List[AtomicMemoryUnit], bases: List[BaseMemoryUnit]) -> str:
        text = "[EVIDENCE FACTS (Abstracted Insights)]:\n"
        if not atomics:
            text += "No specific facts extracted.\n"
        for i, fact in enumerate(atomics):
            ts_str = ""
            if fact.metadata.get("timestamp"):
                ts_str = f"[Message Date: {str(fact.metadata.get('timestamp'))[:16]}] "
            text += f"{i + 1}. {fact.fact_text}{ts_str}\n"

        text += "\n[SOURCE CONVERSATION (Original Context)]:\n"
        bases.sort(key=lambda x: str(x.timestamp) if x.timestamp else "")
        for base in bases:
            speaker = base.metadata.get("speaker", "Unknown")
            ts = str(base.timestamp)[:16] if base.timestamp else ""
            text += f"- [{ts}] {speaker}: {base.text}\n"

        text += "\n[INSTRUCTIONS FOR EVIDENCE PROCESSING]:\n"
        text += "1. Prioritize evidence completeness and answer using the retrieved facts.\n"
        text += "2. Use the source conversation to recover exact names and specific details.\n"
        text += "3. Pay attention to timestamps and temporal reasoning when answering.\n"
        text += "4. Combine multiple facts when the question requires synthesis.\n"
        return text

    def _build_final_prompt(self, question: str, evidence: str, query_date: str = None) -> str:
        time_instruction = ""
        if query_date:
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
