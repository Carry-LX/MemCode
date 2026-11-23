import asyncio
import json
import re
from datetime import datetime
from typing import List, Any, Dict, Optional
from src.core.memory_structures import BaseMemoryUnit, AtomicMemoryUnit


class AsyncAbstractionEngine:
    """
    异步抽象引擎 - 事实提取 (支持精确时间版)
    """

    def __init__(self, llm_interface: Any, max_concurrent: int = 16):

        self.llm = llm_interface
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _is_factual_sentence(self, sentence: str) -> bool:


        return True

    def _chunk_contains_factual_content(self, chunk: List[BaseMemoryUnit]) -> bool:

        return True

    def _build_fact_extraction_prompt(self, context: str, has_context: bool = False) -> str:
        # Context-present prompt: background resolves references but is not itself extracted.
        if has_context:
            return f"""
                You are an expert Memoryist. Extract dense, stand-alone facts from the conversation.

                RULES:
1. **Filter Noise**: IGNORE greetings, short reactions, and vague chitchat.
2. **Resolve Identity**: REPLACE pronouns (I/you/he) with actual Speaker Names.
3. **Time Rewriting (CRITICAL)**:
   - You MUST rewrite relative time terms (e.g., "yesterday", "last week") into EXACT DATES/TIMES.
   - Use the "--- Timestamp: YYYY-MM-DD HH:MM ---" header above the text as the anchor.
   - **Example**:
     Context: "--- Timestamp: 2023-05-10 14:30 ---\\n[User]: I just arrived."
     Output Fact: "User arrived at 14:30 on 2023-05-10."
4. **Merge**: Combine fragmented sentences into one complete fact.
5. **Context**: If a line is marked as [Background], use it for understanding but DO NOT extract facts from it.

                Input Context:
                {context}

                Output strictly JSON: [{{"fact": "SpeakerName action...", "source_indices": [1, 2]}}]
                """

        # Context-free prompt: extract only the numbered target messages.
        else:
            return f"""
                You are an expert Memoryist. Extract dense, stand-alone facts from the conversation.

                RULES:
1. **Filter Noise**: IGNORE greetings, short reactions, and vague chitchat.
2. **Resolve Identity**: REPLACE pronouns (I/you/he) with actual Speaker Names.
3. **Time Rewriting (CRITICAL)**:
   - You MUST rewrite relative time terms (e.g., "yesterday", "last week") into EXACT DATES/TIMES.
   - Use the "--- Timestamp: YYYY-MM-DD HH:MM ---" header above the text as the anchor.
   - **Example**:
     Context: "--- Timestamp: 2023-05-10 14:30 ---\\n[User]: I just arrived."
     Output Fact: "User arrived at 14:30 on 2023-05-10."
4. **Merge**: Combine fragmented sentences into one complete fact.

                Input Context:
                {context}

                Output strictly JSON: [{{"fact": "SpeakerName action...", "source_indices": [1, 2]}}]
                """

    # 关联判断
    def _build_linking_prompt(self, new_fact: str, new_date: str, old_mems: List[AtomicMemoryUnit]) -> str:
        candidates_str = ""
        for i, m in enumerate(old_mems):
            # 这里顺便也展示更详细一点的时间，辅助关联
            ts = str(m.metadata.get('timestamp', 'Unknown'))[:16] # 取到分钟
            candidates_str += f"[{i}] [Date:{ts}] {m.fact_text}\n"

        return f"""
        Determine if the NEW FACT is LOGICALLY CONNECTED to any EXISTING FACTS.
        Only link if they describe the **same event/storyline** or have a clear **cause-effect**.

        [NEW FACT] ({new_date}): {new_fact}
        [EXISTING FACTS]:
        {candidates_str}

        Output strictly JSON: {{"related_indices": [0, 2]}} or {{"related_indices": []}}
        """

    async def identify_relations_async(self, new_mem: AtomicMemoryUnit, candidate_mems: List[AtomicMemoryUnit]) -> List[
        str]:
        if not candidate_mems: return []
        async with self.semaphore:
            return await asyncio.get_event_loop().run_in_executor(None, self._identify_relations_sync, new_mem,
                                                                  candidate_mems)

    def _identify_relations_sync(self, new_mem: AtomicMemoryUnit, candidate_mems: List[AtomicMemoryUnit]) -> List[str]:
        ts_new = str(new_mem.metadata.get('timestamp', 'Unknown'))[:16]
        prompt = self._build_linking_prompt(new_mem.fact_text, ts_new, candidate_mems)
        response = self.llm.get_completion(prompt)
        linked_ids = []
        try:
            cleaned = response.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned)
            indices = data.get("related_indices", [])
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(candidate_mems):
                    linked_ids.append(candidate_mems[idx].id)
        except:
            pass
        return linked_ids


    def create_extraction_tasks(self, chunks: List[List[BaseMemoryUnit]]) -> List[asyncio.Task]:
        tasks = []
        if not chunks: return tasks

        for chunk in chunks:
            if chunk and self._chunk_contains_factual_content(chunk):
                task = asyncio.create_task(self._extract_single_chunk_async(chunk))
                tasks.append(task)

        return tasks

    async def extract_atomic_facts_batch_async(self, chunks: List[List[BaseMemoryUnit]]) -> List[
        List[AtomicMemoryUnit]]:
        tasks = self.create_extraction_tasks(chunks)
        if not tasks: return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for res in results:
            if isinstance(res, Exception):
                print(f"[Engine Error] Extraction failed: {res}")
                final_results.append([])
            else:
                final_results.append(res)
        return final_results

    async def _extract_single_chunk_async(self, chunk: List[BaseMemoryUnit]) -> List[AtomicMemoryUnit]:
        async with self.semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._extract_facts_sync, chunk
            )

    def _extract_facts_sync(self, chunk: List[BaseMemoryUnit]) -> List[AtomicMemoryUnit]:
        """同步提取逻辑"""
        if not chunk: return []

        factual_lines = []
        index_to_parent_id_map = {}
        index_to_timestamp_map = {}

        msg_counter = 0
        last_time_str = None

        # Detect context from message metadata rather than dataset name.
        has_context_in_batch = any(mem.metadata.get('is_context', False) for mem in chunk)

        for i, mem in enumerate(chunk):
            if self._is_factual_sentence(mem.text):
                msg_counter += 1
                speaker = mem.metadata.get('speaker', 'Unknown')

                # 仅当检测到且确实是上下文时，才添加前缀
                is_context = mem.metadata.get('is_context', False)
                prefix = "[Background] " if is_context else ""

                # 获取时间
                ts_obj = getattr(mem, 'timestamp', None) or mem.metadata.get('original_timestamp')
                current_time_str = "Unknown Time"

                # 格式化为 YYYY-MM-DD HH:MM
                if ts_obj:
                    if isinstance(ts_obj, datetime):
                        current_time_str = ts_obj.strftime('%Y-%m-%d %H:%M')
                    else:
                        # 尝试保留前16位 (YYYY-MM-DD HH:MM)
                        s_ts = str(ts_obj)
                        current_time_str = s_ts[:16] if len(s_ts) >= 16 else s_ts

                # 插入精确时间头
                if current_time_str != last_time_str:
                    factual_lines.append(f"\n--- Timestamp: {current_time_str} ---")
                    last_time_str = current_time_str

                # Format numbered target messages with their timestamp and speaker.
                factual_lines.append(f"[{msg_counter}] {prefix}[{speaker}]: {mem.text}")

                parent_id = mem.metadata.get('parent_bmu_id')
                if parent_id:
                    index_to_parent_id_map[msg_counter] = parent_id

                if ts_obj:
                    index_to_timestamp_map[msg_counter] = ts_obj

        if not factual_lines:
            return []

        # 调用 LLM
        context_str = "\n".join(factual_lines)

        # 传入 has_context 状态，触发 explicit if/else
        prompt = self._build_fact_extraction_prompt(context_str, has_context=has_context_in_batch)


        response_str = self.llm.get_completion(prompt)
        if not response_str: return []

        # 解析结果
        return self._parse_llm_json(response_str, index_to_parent_id_map, index_to_timestamp_map)

    def _parse_llm_json(self, response_str: str, id_map: Dict, ts_map: Dict) -> List[AtomicMemoryUnit]:
        """辅助函数：解析和清理 JSON"""
        cleaned = response_str.strip()
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r'("fact":\s*)([^"\s\[{])', r'\1"\2', cleaned)
                data = json.loads(fixed)
            except:
                return []

        results = []
        if isinstance(data, list):
            for item in data:
                fact = item.get("fact")
                indices = item.get("source_indices")

                if indices and not isinstance(indices, list):
                    indices = [indices]

                if fact and indices:
                    p_ids = list({id_map[idx] for idx in indices if idx in id_map})
                    t_stamp = None
                    for idx in indices:
                        if idx in ts_map:
                            t_stamp = ts_map[idx]
                            break

                    meta = {}
                    if t_stamp: meta['timestamp'] = t_stamp

                    if p_ids:
                        results.append(AtomicMemoryUnit(
                            fact_text=fact,
                            source_base_ids=p_ids,
                            metadata=meta
                        ))
        return results
