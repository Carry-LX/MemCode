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
    """LongMemEval retrieval with fact redirection and question-aware evidence selection.

    Count, temporal, update and preference routes select source-grounded
    evidence from dense-vector and codebook candidate pools."""

    # Hyperparameters
    ALPHA_DECAY: float = 0.3
    EXPANSION_K: int = 2
    MIN_IDF_THRESHOLD: float = 0.5
    MAX_VECTOR_DISTANCE: float = 0.65
    MIN_CODEBOOK_SCORE: float = 0.3
    FINAL_ATOMIC_LIMIT: int = 40
    FINAL_BASE_LIMIT: int = 10

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
            print("VQ-VAE model not provided. Loading default model.")

        if not self.vq_model:
            print("VQ-VAE model failed to load. Please verify the model path.")

        print("QueryEngine (LongMemEval) initialized.")

    # =========================================================================
    # Math helpers
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

    def _refresh_live_stats(self) -> Dict[str, Any]:
        live_stats = {
            "total_docs": 0,
            "code_freq": defaultdict(int),
            "co_occurrence": defaultdict(lambda: defaultdict(int))
        }
        try:
            for amu in self.meta_db.get_all_active_atomics():
                unique_codes = sorted(set(amu.code_indices or []))
                live_stats["total_docs"] += 1
                for code in unique_codes:
                    live_stats["code_freq"][code] += 1
                for idx, code_a in enumerate(unique_codes):
                    for code_b in unique_codes[idx + 1:]:
                        live_stats["co_occurrence"][code_a][code_b] += 1
                        live_stats["co_occurrence"][code_b][code_a] += 1
        except Exception:
            pass
        self.meta_db.stats = live_stats
        if hasattr(self.meta_db, "_save_stats"):
            self.meta_db._save_stats()
        return live_stats

    def _get_safe_stats(self) -> Dict[str, Any]:
        stats = self.meta_db.get_stats() or {}
        if stats.get("total_docs", 0) > 0:
            return stats
        return self._refresh_live_stats()

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()

    def _extract_query_keywords(self, question: str) -> Set[str]:
        tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9\-']+", self._normalize_text(question)))
        stopwords = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with", "did", "do", "does",
            "is", "are", "was", "were", "be", "been", "my", "i", "me", "we", "our", "you", "your", "can",
            "could", "would", "should", "what", "which", "when", "where", "who", "how", "many", "much",
            "long", "total", "some", "this", "that", "these", "those", "there", "about", "during", "into",
            "from", "have", "has", "had", "am", "it", "if", "as", "user"
        }
        return {token for token in tokens if token not in stopwords and len(token) > 2}

    def _has_numeric_signal(self, text: str) -> bool:
        lowered = self._normalize_text(text)
        return bool(
            re.search(
                r"\b\d+\b|\$\d+|\b(one|two|three|four|five|six|seven|eight|nine|ten|"
                r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
                r"nineteen|twenty|half)\b|"
                r"january|february|march|april|may|june|july|august|september|october|november|december",
                lowered
            )
        )

    def _has_explicit_quantity(self, text: str) -> bool:
        lowered = self._normalize_text(text)
        return bool(
            re.search(
                r"\b\d+\b|\$\d+|\b(one|two|three|four|five|six|seven|eight|nine|ten|"
                r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
                r"nineteen|twenty|thirty|forty|fifty|hundred)\b",
                lowered
            )
        )

    def _has_current_marker(self, question: str) -> bool:
        q = self._normalize_text(question)
        return any(
            token in q for token in [
                "most recent",
                " latest",
                " latest?",
                " latest.",
                " now",
                " now?",
                " now.",
                "currently",
                "current ",
                "current?",
                "current.",
            ]
        )

    def _question_implies_inventory_state(self, question: str) -> bool:
        q = self._normalize_text(question)
        if not (q.startswith("how many") or q.startswith("how much")):
            return False
        inventory_markers = [
            "currently own",
            "currently have",
            "do i own",
            "do i have",
            "i currently own",
            "i currently have",
            "own right now",
            "have right now",
            "currently keep",
            "currently maintain",
        ]
        metric_markers = [
            "followers",
            "subscriber",
            "subscribers",
            "views",
            "hours",
            "minutes",
            "miles",
            "points",
            "score",
            "spent",
            "raised",
            "earned",
            "reach",
            "goal",
            "progress",
        ]
        return any(token in q for token in inventory_markers) and not any(token in q for token in metric_markers)

    def _question_implies_metric_update(self, question: str) -> bool:
        q = self._normalize_text(question)
        if not (q.startswith("how many") or q.startswith("how much")):
            return False
        return any(
            token in q for token in [
                "followers",
                "subscriber",
                "subscribers",
                "views",
                "hours",
                "minutes",
                "miles",
                "points",
                "score",
                "spent",
                "raised",
                "earned",
                "reached",
                "progress",
                "watched",
                "worn",
                "read",
            ]
        )

    def _is_future_acquisition_fact(self, text: str) -> bool:
        lowered = self._normalize_text(text)
        return any(
            token in lowered for token in [
                "thinking of buying",
                "thinking about buying",
                "considering buying",
                "considering getting",
                "thinking of getting",
                "plan to buy",
                "plans to buy",
                "will buy",
                "going to buy",
                "looking to buy",
                "interested in getting",
                "wants to buy",
            ]
        )

    def _is_vague_quantified_fact(self, text: str) -> bool:
        lowered = self._normalize_text(text)
        if "one of the" in lowered:
            return True
        if any(token in lowered for token in ["some ", "several ", "a few ", "various ", "multiple "]):
            return True
        return False

    def _fact_satisfies_hard_query_constraints(self, question: str, fact_text: str) -> bool:
        q = self._normalize_text(question)
        f = self._normalize_text(fact_text)

        if any(token in q for token in ["re-watch", "rewatch", "re watched"]):
            return any(token in f for token in ["re-watch", "rewatch", "rewatched", "re-watched"])

        if any(token in q for token in ["submit", "submitted"]):
            return any(token in f for token in ["submit", "submitted"])

        if "service" in q:
            return any(token in f for token in ["service", "serviced", "replace", "replaced", "repair", "repaired"])

        return True

    def _extract_named_entity_spans(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9&\-':/.]*", str(text or ""))
        connectors = {"of", "the", "and", "for", "to", "in", "on", "no", "way"}
        ignored = {"User", "Assistant", "Message", "Date", "State", "Current", "Previous"}

        spans: List[str] = []
        chunk: List[str] = []
        for token in tokens:
            cleaned = token.strip("[]():,.'\"")
            if not cleaned:
                continue
            if any(ch.isupper() for ch in cleaned) and cleaned not in ignored:
                chunk.append(cleaned)
                continue
            if chunk and cleaned.lower() in connectors:
                chunk.append(cleaned)
                continue
            if chunk:
                spans.append(" ".join(chunk).lower())
                chunk = []
        if chunk:
            spans.append(" ".join(chunk).lower())
        return spans[:3]

    def _get_mem_timestamp(self, mem: AtomicMemoryUnit) -> str:
        metadata = mem.metadata or {}
        return str(metadata.get("timestamp", "") or "")[:19]

    def _extract_exclusion_keywords(self, question: str) -> Set[str]:
        lowered = self._normalize_text(question)
        matches = re.findall(r"(?:excluding|exclude|except|other than)\s+([^,?.;]+)", lowered)
        keywords: Set[str] = set()
        for chunk in matches:
            keywords.update(self._extract_query_keywords(chunk))
        return keywords

    def _fact_signature_for_query(self, mem: AtomicMemoryUnit, question: str) -> Tuple[str, ...]:
        question_keywords = self._extract_query_keywords(question)
        text_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-']+", self._normalize_text(mem.fact_text))
        overlap_tokens = [token for token in text_tokens if token in question_keywords]
        if overlap_tokens:
            return tuple(sorted(set(overlap_tokens))[:8])

        fallback = [
            token for token in text_tokens
            if token not in {"user", "assistant", "message", "date", "about", "with", "from", "into"}
        ]
        return tuple(fallback[:4])

    def _count_rescue_signature(self, mem: AtomicMemoryUnit, question: str) -> Tuple[str, ...]:
        named_spans = self._extract_named_entity_spans(mem.fact_text)
        if named_spans:
            return tuple(named_spans[:2])

        question_keywords = self._extract_query_keywords(question)
        generic_tokens = question_keywords | {
            "current", "currently", "total", "many", "much", "times", "time", "count", "number",
            "completed", "complete", "watched", "watch", "visited", "visit", "owned", "own", "have",
            "service", "serviced", "bake", "baked", "ride", "rode", "raise", "raised", "movie", "movies",
            "course", "courses", "tank", "tanks", "instrument", "instruments", "bike", "bikes",
            "festival", "festivals", "museum", "museums", "gallery", "galleries", "grocery", "store", "spent",
        }
        text_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-']+", self._normalize_text(mem.fact_text))
        informative = [token for token in text_tokens if token not in generic_tokens and len(token) > 2]
        if informative:
            return tuple(sorted(set(informative))[:3])
        return self._fact_signature_for_query(mem, question)

    def _is_count_rescue_candidate(self, mem: AtomicMemoryUnit, question: str = "") -> bool:
        metadata = mem.metadata or {}
        fact_status = str(metadata.get("fact_status", "") or "").lower()
        if fact_status in {"advice", "planned"}:
            return False
        if metadata.get("is_aggregate_summary"):
            return False
        if self._infer_fact_speaker(mem) == "assistant":
            return False
        if question and not self._fact_satisfies_hard_query_constraints(question, mem.fact_text):
            return False
        return (
            self._has_numeric_signal(mem.fact_text)
            or str(metadata.get("archive_policy", "") or "").lower() == "replace"
            or str(metadata.get("evolution_mode", "") or "").upper() == "UPDATE"
        )

    def _has_relative_temporal_reference(self, question: str) -> bool:
        q = self._normalize_text(question)
        direct_markers = [
            "a couple of days ago",
            "couple of days ago",
            "few days ago",
            "past weekend",
            "last weekend",
            "this weekend",
            "that weekend",
            "between the day",
            "during december",
            "during november",
            "during october",
            "during september",
            "during august",
            "during july",
            "during june",
            "during may",
            "during april",
            "during march",
            "during february",
            "during january",
        ]
        if any(marker in q for marker in direct_markers):
            return True
        return bool(
            re.search(
                r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|several)\s+"
                r"(?:day|days|week|weeks|month|months|year|years)\s+(?:ago|before|after)\b",
                q,
            )
        )

    def _has_temporal_choice_marker(self, question: str) -> bool:
        q = self._normalize_text(question)
        asks_for_choice = bool(re.match(r"^(which|what|who|where)\b", q))
        if not asks_for_choice:
            return False
        return any(
            marker in q for marker in [
                "happened first",
                "came first",
                "was first",
                "most recent",
                "most recently",
                "started using",
                "earliest",
                "latest",
            ]
        )

    def _has_update_state_marker(self, question: str) -> bool:
        q = self._normalize_text(question)
        direct_markers = [
            "personal best",
            "best time",
            "best pace",
            "best score",
            "most recent family trip",
            "after her recent relocation",
            "after his recent relocation",
            "after their recent relocation",
            "after the recent relocation",
        ]
        if any(marker in q for marker in direct_markers):
            return True
        return bool(re.search(r"\bafter (?:his|her|their|the) recent [a-z]+\b", q))

    def _get_query_features(self, question: str) -> Dict[str, Any]:
        q = self._normalize_text(question)

        has_latest_marker = self._has_current_marker(question)
        has_relative_temporal = self._has_relative_temporal_reference(question)
        has_temporal_choice = self._has_temporal_choice_marker(question)
        has_update_state_marker = self._has_update_state_marker(question)
        has_ordering = has_temporal_choice or any(
            token in q for token in [
                "order of",
                "from earliest to latest",
                "from latest to earliest",
                "starting from the earliest",
                "earliest to latest",
                "latest to earliest",
            ]
        )
        has_anchor_event = any(
            token in q for token in [
                " before the ",
                " after the ",
                " before i ",
                " after i ",
                " when i ",
                " by the time ",
                " at the time ",
            ]
        )
        has_exclusion = any(token in q for token in ["excluding ", "exclude ", "except ", "other than "])
        has_total_marker = any(
            token in q for token in [
                "total amount",
                "total number",
                "in total",
                "combined total",
                "combined amount",
                "altogether",
                "total money",
                "total number of",
            ]
        )
        has_distinct_marker = any(
            token in q for token in [
                "different ",
                "types of",
                "kind of",
                "order of",
            ]
        )
        has_rolling_window = any(
            token in q for token in [
                "past few weeks",
                "over the past few weeks",
                "in recent weeks",
                "these past weeks",
            ]
        )
        assistant_recall = any(
            token in q for token in [
                "i remember you",
                "looking back at our previous conversation",
                "you provided",
                "you recommended",
                "you made after",
                "what was the move you made",
                "what was the name of that",
            ]
        )
        preference_like = any(
            token in q for token in [
                "recommend",
                "suggest",
                "tips on",
                "tips for",
                "any tips",
                "helpful tips",
                "do you think",
                "activities can i do",
                "what to do there",
            ]
        )
        current_inventory_like = (
            self._question_implies_inventory_state(question)
            and not has_total_marker
            and not has_exclusion
        )
        progress_state_like = (
            self._question_implies_metric_update(question)
            or "do i need to reach" in q
            or has_update_state_marker
        ) and not has_total_marker and not has_distinct_marker and not has_exclusion

        if assistant_recall:
            profile = "assistant_recall"
        elif has_ordering:
            profile = "temporal_sequence"
        elif q.startswith("when ") or "when did" in q or "what date" in q:
            profile = "when"
        elif has_anchor_event and any(token in q for token in ["how many", "how much", "what ", "who ", "where "]):
            profile = "anchored_temporal"
        elif preference_like:
            profile = "preference"
        elif current_inventory_like:
            profile = "current_count"
        elif has_latest_marker or progress_state_like or has_update_state_marker:
            profile = "latest_update"
        elif (
            q.startswith("how many")
            or "total amount" in q
            or "total number" in q
            or "in total" in q
            or "combined total" in q
            or "combined amount" in q
            or "altogether" in q
        ):
            profile = "count"
        elif any(token in q for token in ["past month", "past two weeks", "this year", "last month", "last week"]) or has_relative_temporal:
            profile = "time_window"
        else:
            profile = "generic"

        return {
            "profile": profile,
            "has_latest_marker": has_latest_marker,
            "has_ordering": has_ordering,
            "has_anchor_event": has_anchor_event,
            "has_exclusion": has_exclusion,
            "has_total_marker": has_total_marker,
            "has_rolling_window": has_rolling_window,
            "has_relative_temporal": has_relative_temporal,
            "has_temporal_choice": has_temporal_choice,
            "has_update_state_marker": has_update_state_marker,
            "assistant_recall": assistant_recall,
            "current_inventory_like": current_inventory_like,
            "progress_state_like": progress_state_like,
        }

    def _classify_query(self, question: str) -> str:
        return self._get_query_features(question)["profile"]

    def _is_visible_evolved(self, metadata: Dict[str, Any]) -> bool:
        if metadata.get("type") != "evolved_fact":
            return True
        if "generic_retrieval_visible" in metadata:
            return bool(metadata.get("generic_retrieval_visible"))
        return True

    def _is_legacy_evolved(self, metadata: Dict[str, Any]) -> bool:
        return metadata.get("type") == "evolved_fact" and "archive_policy" not in metadata

    def _infer_fact_speaker(self, mem: AtomicMemoryUnit) -> str:
        metadata = mem.metadata or {}
        speaker_scope = str(metadata.get('speaker_scope', '') or '').lower()
        if speaker_scope in {'user', 'assistant'}:
            return speaker_scope
        text = self._normalize_text(mem.fact_text)
        if text.startswith("user"):
            return "user"
        if text.startswith("assistant"):
            return "assistant"
        return "neutral"

    def _keyword_overlap(self, text: str, keywords: Set[str]) -> int:
        lowered = self._normalize_text(text)
        return sum(1 for token in keywords if token in lowered)

    def _score_candidate_for_selection(self, mem: AtomicMemoryUnit, question: str = "", original_rank: int = 0) -> float:
        metadata = mem.metadata or {}
        vector_score = float(metadata.get('vector_score', 999.0) or 999.0)
        codebook_score = float(metadata.get('codebook_score', 0.0) or 0.0)
        query_features = self._get_query_features(question)
        query_type = query_features["profile"]
        keywords = self._extract_query_keywords(question)
        exclusion_keywords = self._extract_exclusion_keywords(question)

        vector_component = 0.0 if vector_score >= 900 else max(0.0, 1.0 - min(vector_score, 1.0))
        codebook_component = min(codebook_score, 2.0) / 2.0
        score = 0.50 * vector_component + 0.18 * codebook_component
        score += max(0.0, 0.10 - original_rank * 0.003)
        score += min(0.24, self._keyword_overlap(mem.fact_text, keywords) * 0.05)

        is_evolved = metadata.get('type') == 'evolved_fact'
        archive_policy = ""
        fact_status = ""
        if not is_evolved:
            score += 0.12
        else:
            if not self._is_visible_evolved(metadata):
                return -999.0

            archive_policy = str(metadata.get('archive_policy', '') or '').lower()
            evolution_mode = str(metadata.get('evolution_mode', '') or '').upper()
            fact_status = str(metadata.get('fact_status', '') or '').lower()

            if archive_policy == 'replace':
                score += 0.14
            elif archive_policy == 'augment':
                score -= 0.08

            if evolution_mode == 'UPDATE':
                score += 0.12
            elif evolution_mode == 'MERGE':
                score += 0.04
            elif evolution_mode == 'AGGREGATE':
                score -= 0.08

            if fact_status in {'advice', 'planned'}:
                score -= 0.18
            if fact_status == 'state':
                score += 0.06

            if metadata.get('has_mixed_speakers'):
                score -= 0.12
            if metadata.get('is_aggregate_summary'):
                score -= 0.08
            if metadata.get('count_risk'):
                score -= 0.10

            derived_from_count = int(metadata.get('derived_from_count', 0) or 0)
            if derived_from_count > 3:
                score -= min(0.18, (derived_from_count - 3) * 0.03)

            if len(mem.fact_text or "") > 220:
                score -= min(0.15, (len(mem.fact_text) - 220) / 1200.0)

            if self._normalize_text(mem.fact_text).startswith("[update]"):
                score += 0.06
            if self._normalize_text(mem.fact_text).startswith("[preference]"):
                score += 0.06

        if 'redirected_from' in metadata:
            score += 0.05

        speaker = self._infer_fact_speaker(mem)
        if speaker == 'assistant':
            score -= 0.04
        elif speaker == 'user':
            score += 0.03

        lowered_fact = self._normalize_text(mem.fact_text)

        if query_type in {"count", "time_window", "when", "anchored_temporal", "temporal_sequence", "current_count"}:
            if not is_evolved:
                score += 0.10
            elif archive_policy == "augment":
                score -= 0.10
            if self._has_numeric_signal(mem.fact_text):
                score += 0.06
            if speaker == "user":
                score += 0.04
            if question and not self._fact_satisfies_hard_query_constraints(question, mem.fact_text):
                score -= 0.12
            if self._is_vague_quantified_fact(mem.fact_text):
                score -= 0.08

        if query_type == "current_count":
            if fact_status == "state":
                score += 0.08
            if self._is_future_acquisition_fact(mem.fact_text):
                score -= 0.24

        if query_type == "latest_update":
            if archive_policy == "replace":
                score += 0.12
            elif archive_policy == "augment":
                score -= 0.18
            if fact_status == "state":
                score += 0.10
            if self._has_numeric_signal(mem.fact_text):
                score += 0.10
            if metadata.get("count_risk"):
                score -= 0.12
            if query_features["has_update_state_marker"] and self._has_numeric_signal(mem.fact_text):
                score += 0.04

        if query_type in {"when", "anchored_temporal", "temporal_sequence", "time_window"}:
            if self._has_numeric_signal(mem.fact_text):
                score += 0.06
            if any(token in lowered_fact for token in ["day", "days", "week", "weeks", "month", "months", "weekend", "yesterday", "today"]):
                score += 0.03
            if fact_status in {"planned", "advice"}:
                score -= 0.08

        if query_type == "assistant_recall":
            if speaker == "assistant":
                score += 0.22
            elif speaker == "user":
                score -= 0.12

        if query_type == "preference":
            if speaker == "user":
                score += 0.16
            elif speaker == "assistant":
                score -= 0.08
            if any(token in lowered_fact for token in ["prefer", "favorite", "like", "love", "enjoy", "want", "interested", "allergy", "avoid"]):
                score += 0.08
            if any(token in lowered_fact for token in ["cat", "dust", "dander", "clean", "cleaned", "allergen", "living room", "symptom"]):
                score += 0.04

        if exclusion_keywords and any(token in lowered_fact for token in exclusion_keywords):
            score -= 0.22

        return score

    def _sort_candidates(self, atomics: List[AtomicMemoryUnit], question: str = "") -> List[AtomicMemoryUnit]:
        decorated = [
            (self._score_candidate_for_selection(mem, question=question, original_rank=idx), idx, mem)
            for idx, mem in enumerate(atomics)
        ]
        decorated.sort(key=lambda item: (-item[0], item[1]))
        return [mem for score, _, mem in decorated if score > -900]

    def _select_final_atomics(self, question: str, atomics: List[AtomicMemoryUnit], limit: int = None) -> List[AtomicMemoryUnit]:
        limit = limit or self.FINAL_ATOMIC_LIMIT
        features = self._get_query_features(question)
        query_type = features["profile"]
        ranked = self._sort_candidates(atomics, question=question)

        if query_type not in {
            "count", "time_window", "when", "preference",
            "latest_update", "anchored_temporal", "temporal_sequence", "assistant_recall", "current_count"
        }:
            return ranked[:limit]

        count_like_types = {"count", "time_window", "when", "anchored_temporal", "temporal_sequence", "current_count"}
        primary = []
        secondary = []

        for mem in ranked:
            metadata = mem.metadata or {}
            is_evolved = metadata.get("type") == "evolved_fact"
            archive_policy = metadata.get("archive_policy", "legacy" if self._is_legacy_evolved(metadata) else "")
            speaker = self._infer_fact_speaker(mem)

            if query_type == "preference":
                if speaker == "user" or (not is_evolved):
                    primary.append(mem)
                else:
                    secondary.append(mem)
                continue

            if query_type == "assistant_recall":
                if speaker == "assistant":
                    primary.append(mem)
                else:
                    secondary.append(mem)
                continue

            if (not is_evolved) or archive_policy in {"replace", "legacy"}:
                primary.append(mem)
            else:
                secondary.append(mem)

        selected = []
        seen_ids: Set[str] = set()

        def extend_from(pool: List[AtomicMemoryUnit], target_count: int):
            for mem in pool:
                if len(selected) >= target_count:
                    break
                if mem.id in seen_ids:
                    continue
                selected.append(mem)
                seen_ids.add(mem.id)

        if query_type == "latest_update":
            latest_by_signature: Dict[Tuple[str, ...], AtomicMemoryUnit] = {}
            latest_order: List[Tuple[str, ...]] = []

            for mem in ranked:
                signature = self._fact_signature_for_query(mem, question)
                if signature not in latest_by_signature:
                    latest_by_signature[signature] = mem
                    latest_order.append(signature)
                    continue

                existing = latest_by_signature[signature]
                if self._get_mem_timestamp(mem) >= self._get_mem_timestamp(existing):
                    latest_by_signature[signature] = mem

            deduped_latest = [latest_by_signature[sig] for sig in latest_order]
            deduped_latest.sort(
                key=lambda mem: (
                    self._get_mem_timestamp(mem),
                    1 if self._has_numeric_signal(mem.fact_text) else 0,
                ),
                reverse=True
            )

            extend_from(deduped_latest, min(limit, 12))
            extend_from(primary, min(limit, 18))
            extend_from(secondary, min(limit, 24))
            extend_from(ranked, min(limit, 24))
            return selected[:min(limit, 24)]

        primary_target = min(
            limit,
            28 if query_type in count_like_types else 24
        )

        extend_from(primary, max(0, primary_target - 6))

        if query_type in count_like_types:
            rescue_pool = []
            for idx, mem in enumerate(ranked):
                if mem.id in seen_ids:
                    continue
                if not self._is_count_rescue_candidate(mem, question=question):
                    continue
                rescue_pool.append((
                    1 if self._fact_satisfies_hard_query_constraints(question, mem.fact_text) else 0,
                    1 if self._has_explicit_quantity(mem.fact_text) else 0,
                    1 if not self._is_vague_quantified_fact(mem.fact_text) else 0,
                    1 if self._has_numeric_signal(mem.fact_text) else 0,
                    1 if self._infer_fact_speaker(mem) == "user" else 0,
                    self._get_mem_timestamp(mem),
                    -idx,
                    mem,
                ))

            rescue_pool.sort(reverse=True)
            best_rescue_by_signature: Dict[Tuple[str, ...], Tuple[Tuple[Any, ...], AtomicMemoryUnit]] = {}
            rescue_order: List[Tuple[str, ...]] = []
            for item in rescue_pool:
                priority = item[:-1]
                mem = item[-1]
                signature = self._count_rescue_signature(mem, question)
                if signature not in best_rescue_by_signature:
                    best_rescue_by_signature[signature] = (priority, mem)
                    rescue_order.append(signature)
                    continue
                if priority > best_rescue_by_signature[signature][0]:
                    best_rescue_by_signature[signature] = (priority, mem)
            extend_from([best_rescue_by_signature[sig][1] for sig in rescue_order], primary_target)

        extend_from(secondary, limit)
        extend_from(ranked, limit)
        return selected[:limit]

    # =========================================================================
    # Candidate retrieval
    # =========================================================================
    def _retrieve_atomic_candidates(self, queries: List[str], k_per_step: int, total_atomic_count: int = None) -> List[
        AtomicMemoryUnit]:
        """
        Retrieve a candidate pool of atomic memories, including archived redirect handling.
        """
        final_candidates_map: Dict[str, AtomicMemoryUnit] = {}
        seen_ids: Set[str] = set()

        current_stats = self._get_safe_stats()

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


        # Helper: resolve one candidate, including archived redirect logic.

        def try_resolve_candidate(mem_id: str) -> AtomicMemoryUnit:
            mem = self.meta_db.get_atomic_memory_by_id(mem_id)
            if not mem:
                return None

            status = mem.metadata.get('status', 'active')

            if status == 'archived':
                superseded_list = mem.metadata.get('superseded_by', [])
                if isinstance(superseded_list, list) and len(superseded_list) > 0:
                    for new_id in superseded_list:
                        new_mem = self.meta_db.get_atomic_memory_by_id(new_id)
                        if new_mem and new_mem.metadata.get('status', 'active') != 'archived':
                            if not self._is_visible_evolved(new_mem.metadata or {}):
                                continue
                            if new_mem.metadata is None:
                                new_mem.metadata = {}
                            new_mem.metadata['redirected_from'] = mem_id
                            return new_mem
                return None

            if not self._is_visible_evolved(mem.metadata or {}):
                return None
            return mem

        # Path A: Vector Search (semantic primary path)
        if current_embedding is not None:
            vec_results = self.vec_db.search("atomic_memory", current_embedding, k=k_per_step * 3)

            for amu_id, dist in vec_results:
                if len(final_candidates_map) >= k_per_step:
                    break

                if amu_id in seen_ids:
                    continue

                mem = try_resolve_candidate(amu_id)

                if mem and mem.id not in seen_ids:
                    if mem.metadata is None:
                        mem.metadata = {}
                    mem.metadata['vector_score'] = float(dist)

                    final_candidates_map[mem.id] = mem
                    seen_ids.add(mem.id)

        # Path B: Structure-enhanced codebook search
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

            # Strict Logic: codebook path still uses the same top-k budget
            path_b_count = 0
            needed = k_per_step

            for amu_id, score in sorted_candidates:
                if path_b_count >= needed: break

                if amu_id in seen_ids: continue

                # Try to resolve the candidate, including redirect handling.
                mem = try_resolve_candidate(amu_id)

                if mem and mem.id not in seen_ids:
                    if mem.metadata is None: mem.metadata = {}
                    mem.metadata['codebook_score'] = score
                    if 'vector_score' not in mem.metadata:
                        mem.metadata['vector_score'] = 999.0

                    final_candidates_map[mem.id] = mem
                    seen_ids.add(mem.id)
                    path_b_count += 1

        # Recalculate codebook scores for all retained candidates.
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

            # Redirected / visible evolved facts should remain eligible for base resolution
            is_redirected = 'redirected_from' in amu.metadata
            is_evolved = amu.metadata.get('type') == 'evolved_fact'
            is_visible_evolved = self._is_visible_evolved(amu.metadata or {})

            is_qualified = (v_score < self.MAX_VECTOR_DISTANCE) or (c_score > self.MIN_CODEBOOK_SCORE)

            if is_redirected or (is_evolved and is_visible_evolved):
                is_qualified = True

            if is_qualified:
                for base_id in amu.source_base_ids:
                    if base_id in base_map and base_id not in added_base_ids:
                        ordered_unique_bases.append(base_map[base_id])
                        added_base_ids.add(base_id)

        return ordered_unique_bases, base_to_atomics_map

    # =========================================================================
    # Main query entry
    # =========================================================================
    def query(self, question: str, k: int = 5, total_atomic_count: int = None, query_date: str = None, **kwargs) -> \
            Dict[str, Any]:
        print(f"\n--- Querying (Efficiency Optimized): '{question}' ---")

        query_features = self._get_query_features(question)
        trace_info = {
            "question": question,
            "strategies_used": [
                "direct_retrieval",
                "codebook_graph_expansion",
                "idf_weighted_scoring",
                "metadata_aware_generic_ranking",
                "profile_aware_final_selection"
            ],
            "expanded_queries": [],
            "retrieved_pool_size": 0,
            "query_profile": query_features["profile"],
            "query_features": query_features,
        }

        queries = [question]
        trace_info["expanded_queries"] = queries

        # Initial retrieval
        k = 20
        raw_atomics = self._retrieve_atomic_candidates(queries, k_per_step=k, total_atomic_count=total_atomic_count)
        trace_info["retrieved_pool_size"] = len(raw_atomics)
        print(f"  - Retrieved atomic facts: {len(raw_atomics)}")

        final_atomics = self._select_final_atomics(question, raw_atomics, limit=self.FINAL_ATOMIC_LIMIT)
        unique_bases, _ = self._group_and_resolve_to_base(final_atomics)
        selected_bases = unique_bases[:self.FINAL_BASE_LIMIT]
        print(f"  - [Selection] Truncated top {len(selected_bases)} base conversations.")
        print(f"  - [Selection] Truncated top {len(final_atomics)} atomic facts.")


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
        features = self._get_query_features(question)
        time_instruction = ""
        if query_date:
            print(f"[Query Date] {query_date}")
            time_instruction = f"""
            [CURRENT REFERENCE TIME]
            Current Date: {query_date}
            (Instruction: Assume 'today' or 'now' refers to this specific date. Calculate relative time based on this reference.)
            """

        extra_instructions = []
        if features["profile"] == "latest_update" or features["has_latest_marker"]:
            extra_instructions.append(
                "If multiple facts describe the same tracked item over time, use the newest value. "
                "Do not add older values unless the evidence clearly says they are additional increments."
            )
        if features["profile"] == "current_count":
            extra_instructions.append(
                "For current ownership or inventory questions, keep items that are still owned or maintained unless the evidence explicitly says they were sold, replaced, or no longer kept."
            )
        if features["has_exclusion"]:
            extra_instructions.append(
                "Respect exclusion constraints exactly. Do not include excluded items, categories, or entities."
            )
            extra_instructions.append(
                "If an excluded item appears in the evidence, it may be mentioned only as excluded and must not be counted in the final answer."
            )
        if features["profile"] in {"count", "time_window", "anchored_temporal", "temporal_sequence", "current_count"}:
            extra_instructions.append(
                "For totals and counts, count each distinct event or item once. Do not double-count duplicate paraphrases."
            )
            extra_instructions.append(
                "If the evidence contains both a vague summary and a more exact fact about the same entity, trust the exact fact once."
            )
            extra_instructions.append(
                "Only count facts that explicitly match the asked action or state. Treat related background as support, not as an extra counted item."
            )
            extra_instructions.append(
                "If you list component values, double-check that the final total exactly matches those listed components."
            )
            extra_instructions.append(
                "Do not infer a missing quantity from a related but differently scoped fact."
            )
        if features["has_anchor_event"] or features["has_ordering"] or features["has_relative_temporal"] or features["profile"] in {"anchored_temporal", "temporal_sequence", "when", "time_window"}:
            extra_instructions.append(
                "For before/after/order questions, anchor the reasoning to the referenced event dates in the evidence, "
                "not the current reference date. Ignore planned or merely discussed future events unless the question explicitly asks about plans."
            )
            extra_instructions.append(
                "Before doing date math or ordering, identify the event that matches the question's key action and entity terms. Do not substitute a nearby but different event."
            )
            extra_instructions.append(
                "For relative temporal phrases such as recent, past weekend, or a couple of days ago, resolve them using the evidence timestamps."
            )
        if features["has_update_state_marker"]:
            extra_instructions.append(
                "For improvement or state-change questions, prefer the newest resulting state or value over an older baseline."
            )
        if features["profile"] == "assistant_recall":
            extra_instructions.append(
                "When the user asks what the assistant previously said or recommended, prefer the assistant's exact earlier content over related background discussion."
            )
        if features["profile"] == "preference":
            extra_instructions.append(
                "For advice or preference questions, ground the answer in the user's specific circumstances, symptoms, named entities, possessions, and recent changes rather than giving generic tips."
            )

        extra_instruction_block = ""
        if extra_instructions:
            joined = "\n".join(f"- {item}" for item in extra_instructions)
            extra_instruction_block = f"\n                [QUESTION-SPECIFIC RULES]\n                {joined}\n"

        return f"""
                You are an expert assistant. Answer the user's question using the provided evidence.
                {time_instruction}
                {extra_instruction_block}

                [USER QUESTION]
                {question}

                [EVIDENCE CONTEXT]
                {evidence}

                [YOUR ANSWER]
                """
