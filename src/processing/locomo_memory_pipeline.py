# LoCoMo memory construction with a ten-unit background window.
import copy
import os
import asyncio
import re
import json
import nltk
from typing import List, Dict, Any, Optional
from datetime import datetime

# 导入核心模块
from src.core.memory_structures import BaseMemoryUnit, AtomicMemoryUnit
from src.processing.background_abstraction_engine import AsyncAbstractionEngine


class AsyncProcessingPipeline:
    """
    异步处理管道 - 码本增强通用版 (无摘要层)
    """

    def __init__(self, splitter, embedder, llm, meta_db, vec_db, vq_model_instance=None):

        self.splitter = splitter
        self.embedder = embedder
        self.meta_db = meta_db
        self.vec_db = vec_db
        self.llm = llm

        # 直接使用传入的实例，不再重复加载
        self.vq_model = vq_model_instance
        if self.vq_model is None:
            print("⚠️ VQ-VAE 模型未传入，码本生成将跳过。")

        # 1. 异步抽象引擎
        self.async_engine = AsyncAbstractionEngine(
            llm_interface=self.llm,
            max_concurrent=22
        )

        self.trace_log_dir = os.path.join(os.path.dirname(meta_db.db_path), "extraction_traces")
        print("异步处理管道 (⚠️⚠️⚠️背景加强版⚠️⚠️⚠️) 已初始化。")

    def _parse_locomo_time(self, time_str: str) -> Any:
        """解析 LoCoMo 特有的时间格式"""
        if not time_str or time_str == "Unknown Time": return None
        try:
            return datetime.strptime(time_str.strip(), "%I:%M %p on %d %B, %Y")
        except:
            return datetime.now()

    def _parse_LongMemEval_time(self, time_str: str) -> Any:
        """
        解析 LongMemEval 特有的时间格式
        确保解析出 HH:MM，这是存储详细时间的基础。
        """
        if not time_str or time_str.strip() == "Unknown Time":
            print("⚠️ 无效时间字符串，返回当前时间。")
            return datetime.now()

        clean_str = time_str.strip()
        try:
            # 格式: "2023/05/20 (Sat) 02:21"
            return datetime.strptime(clean_str, "%Y/%m/%d (%a) %H:%M")
        except ValueError:
            try:
                # Fallback: "2023/05/20 02:21"
                return datetime.strptime(clean_str, "%Y/%m/%d %H:%M")
            except ValueError:
                print(f"⚠️ [Time Error] 无法解析时间 '{clean_str}'，返回当前时间。")
                # 生成一个文件代表错误信息
                with open(f"{clean_str}.txt", 'w', encoding='utf-8') as f:
                    f.write(clean_str)
                return datetime.now()

    def export_extraction_trace(self, atomics: List[AtomicMemoryUnit], suffix: str = ""):
        """导出简单的提取日志"""
        if not self.trace_log_dir: return
        if not os.path.exists(self.trace_log_dir):
            os.makedirs(self.trace_log_dir, exist_ok=True)

        trace_data = []
        for amu in atomics:
            trace_data.append({
                "fact": amu.fact_text,
                "codes": amu.code_indices,
                "source_ids": amu.source_base_ids
            })

        safe_suffix = re.sub(r'[\\/*?:"<>|]', "_", str(suffix))
        fpath = os.path.join(self.trace_log_dir, f"trace_{safe_suffix}_{datetime.now().strftime('%H%M%S')}.json")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)


    # =========================================================================
    # 入口 1: LoCoMo 数据集处理
    # 逻辑: 严格话轮划分 (Target) + 完整视野背景注入 (Context Restoration)
    # =========================================================================
    async def process_locomo_data_async(self, conversation_data: Dict, source_id_prefix: str, log_dir: str = None):
        """Process LoCoMo speaker-pair turns with historical background.

        Combine text and image captions. Close a batch when the initiating
        speaker resumes after a response or a third speaker starts a turn."""
        if log_dir: self.trace_log_dir = log_dir

        all_sentences: List[BaseMemoryUnit] = []
        dynamic_batches: List[List[BaseMemoryUnit]] = []

        # [Buffer] 当前正在累积的话轮 (Target: A + B)
        current_batch: List[BaseMemoryUnit] = []

        # [Buffer] 历史对话流 (用于提供 Context)
        history_buffer: List[BaseMemoryUnit] = []

        # [超参数] 上下文窗口大小
        MAX_CONTEXT_WINDOW = 10

        session_keys = [k for k in conversation_data.keys() if re.match(r'^session_\d+$', k)]
        session_keys.sort(key=lambda x: int(x.split('_')[1]))

        print(
            f"\n> LoCoMo驱动: 解析 {len(session_keys)} 个会话片段 (双人对话 A->B 模式 + {MAX_CONTEXT_WINDOW}句Context)...")

        # [状态追踪]
        batch_initiator = None  # 当前轮次的发起者 (A)
        batch_responder = None  # 当前轮次的回应者 (B)

        for s_key in session_keys:
            session_list = conversation_data[s_key]
            time_key = f"{s_key}_date_time"
            raw_time_str = conversation_data.get(time_key, "Unknown Time")
            parsed_timestamp = self._parse_locomo_time(raw_time_str)
            current_source_id = f"{source_id_prefix}_{s_key}"

            for msg in session_list:
                speaker = msg.get('speaker', 'Unknown')
                text = msg.get('text', '')
                caption = msg.get('blip_caption')

                # 1. 图像文本合并处理
                if not text and not caption:
                    continue

                sents = nltk.sent_tokenize(text) if text else []
                if caption:
                    image_tag = f" [Image: {caption}]"
                    if sents:
                        sents[-1] += image_tag
                    else:
                        sents.append(image_tag)

                full_text_for_db = text + (f" [Image: {caption}]" if caption else "")

                # 存 Raw Base
                base = BaseMemoryUnit(
                    text=full_text_for_db, source_document=current_source_id, timestamp=parsed_timestamp,
                    metadata={'speaker': speaker, 'original_timestamp': raw_time_str}
                )
                self.meta_db.add_base_memory(base)

                # 收集当前消息对象
                msg_sentences_buffer = []
                for sent in sents:
                    bmu = BaseMemoryUnit(
                        text=sent, source_document=current_source_id, timestamp=parsed_timestamp,
                        metadata={'parent_bmu_id': base.id, 'speaker': speaker}
                    )
                    all_sentences.append(bmu)
                    msg_sentences_buffer.append(bmu)

                # A -> B 双人对话切分逻辑

                should_split = False

                # 初始化发起者
                if batch_initiator is None:
                    batch_initiator = speaker

                # 判定逻辑：
                if speaker == batch_initiator:
                    # 如果发起者在有了回应者之后又说话了 (A -> B -> A)，视为新的一轮开始
                    if batch_responder is not None:
                        should_split = True
                else:
                    # 如果不是发起者
                    if batch_responder is None:
                        # 确认为回应者 (A -> B)
                        batch_responder = speaker
                    elif speaker != batch_responder:
                        # 如果既不是发起者，也不是当前的回应者 (A -> B -> C)，视为新的一轮
                        should_split = True

                # 执行切分
                if should_split:
                    if current_batch:
                        final_batch = []
                        # 注入 Context
                        if history_buffer:
                            context_source = history_buffer[-MAX_CONTEXT_WINDOW:]
                            for prev_bmu in context_source:
                                ctx_bmu = copy.copy(prev_bmu)
                                ctx_bmu.metadata = prev_bmu.metadata.copy()
                                ctx_bmu.metadata['is_context'] = True
                                final_batch.append(ctx_bmu)

                        # 注入 Target (上一轮完整的 A+B)
                        final_batch.extend(current_batch)
                        dynamic_batches.append(final_batch)

                        # 更新 History
                        history_buffer.extend(current_batch)
                        if len(history_buffer) > 50:
                            history_buffer = history_buffer[-50:]

                    # 重置状态，开始新的一轮 (当前 speaker 成为新的 initiator)
                    current_batch = []
                    batch_initiator = speaker
                    batch_responder = None

                # 将当前消息加入 Buffer (无论是 A 还是 B，都加进来)
                current_batch.extend(msg_sentences_buffer)

        # 3. 处理最后一个遗留的 Batch
        if current_batch:
            final_batch = []
            if history_buffer:
                context_source = history_buffer[-MAX_CONTEXT_WINDOW:]
                for prev_bmu in context_source:
                    ctx_bmu = copy.copy(prev_bmu)
                    ctx_bmu.metadata = prev_bmu.metadata.copy()
                    ctx_bmu.metadata['is_context'] = True
                    final_batch.append(ctx_bmu)

            final_batch.extend(current_batch)
            dynamic_batches.append(final_batch)

        # 4. 送入流式处理
        await self._core_processing_stream(all_sentences, source_suffix="locomo", pre_batched_data=dynamic_batches)


    # =========================================================================
    # 入口 2: LongMemEval 全量处理 (User-Assistant 闭环策略)
    # =========================================================================
    async def process_longmemeval_data_async(self, sessions: List[List[Dict]], timestamps: List[str],
                                             source_id_prefix: str, log_dir: str = None):
        """Process user/assistant turns; close each batch after the assistant responds."""
        if log_dir: self.trace_log_dir = log_dir

        all_sentences: List[BaseMemoryUnit] = []
        dynamic_batches: List[List[BaseMemoryUnit]] = []
        current_batch: List[BaseMemoryUnit] = []

        print(f"\n> LongMemEval驱动: 批量解析 {len(sessions)} 个 Session (按 User-Assistant 闭环划分)...")

        # 遍历所有 Session
        for sess_i, (sess_msgs, ts_str) in enumerate(zip(sessions, timestamps)):
            parsed_timestamp = self._parse_LongMemEval_time(str(ts_str))
            current_source_id = f"{source_id_prefix}_sess_{sess_i}"

            for msg in sess_msgs:
                role = msg.get('role') or msg.get('speaker') or 'unknown'
                content = msg.get('content') or msg.get('text') or ''

                if not content: continue

                # 存 Raw Base
                base = BaseMemoryUnit(
                    text=content,
                    source_document=current_source_id,
                    timestamp=parsed_timestamp,
                    metadata={'speaker': role, 'original_timestamp': str(ts_str)}
                )
                self.meta_db.add_base_memory(base)

                # 分句并加入 Buffer
                msg_sentences_buffer = []
                sents = nltk.sent_tokenize(content)
                for s in sents:
                    sub_base = BaseMemoryUnit(
                        text=s,
                        source_document=current_source_id,
                        timestamp=parsed_timestamp,
                        metadata={'parent_bmu_id': base.id, 'speaker': role}
                    )
                    all_sentences.append(sub_base)
                    msg_sentences_buffer.append(sub_base)


                # 1. 角色判断
                role_lower = str(role).lower()
                is_assistant = role_lower in ['assistant', 'model', 'system']

                # 2. 无论角色是谁，先加入当前 Batch
                # (如果是 User，它会待在这里等 Assistant；如果是 Assistant，它会跟前面的 User 合并)
                current_batch.extend(msg_sentences_buffer)

                # 3. 触发切分条件：只要 Assistant 说完话，这一轮就强制结束。
                # Case A: 之前是 User (User -> Assistant) -> 完美闭环，Flush。
                # Case B: 之前是 Empty (Assistant Start) -> 独立成块，Flush。
                # Case C: 之前是 User -> User (User Monologue -> Assistant) -> 多个 User 合并回答，Flush。
                if is_assistant:
                    dynamic_batches.append(current_batch)
                    current_batch = []

                # 如果是 User 说话，不做操作，继续循环等待下一条消息 (Assistant)

        # 2. 处理最后一个遗留的 Batch (例如以 User 结尾的情况)
        if current_batch:
            dynamic_batches.append(current_batch)

        # 3. 统一送入处理流
        await self._core_processing_stream(all_sentences, source_suffix=source_id_prefix,
                                           pre_batched_data=dynamic_batches)

    # =========================================================================
    # 入口 3: 旧的单 Session 处理
    # =========================================================================
    async def process_session_async(self, session_messages: List[dict], timestamp: str, source_id: str):
        """(Deprecated) 单 Session 处理"""
        await self.process_longmemeval_data_async([session_messages], [timestamp], source_id)

    # =========================================================================
    # 核心处理流
    # =========================================================================
    async def _core_processing_stream(self,
                                      all_sentences: List[BaseMemoryUnit],
                                      source_suffix: str = "",
                                      pre_batched_data: Optional[List[List[BaseMemoryUnit]]] = None):
        if not all_sentences: return

        print(f"> 驱动: 启动并行流水线处理 {len(all_sentences)} 个句子...")

        # 1. 批次准备
        if pre_batched_data:
            batches = pre_batched_data
            print(f"  > 载入 {len(batches)} 个预设批次 (动态分块)")
        else:
            BATCH_SIZE = 15
            batches = [all_sentences[i:i + BATCH_SIZE] for i in range(0, len(all_sentences), BATCH_SIZE)]
            print(f"  > 切分为 {len(batches)} 个批次 (固定Size)")

        # 2. 提交所有 LLM 任务
        llm_tasks = self.async_engine.create_extraction_tasks(batches)
        if not llm_tasks:
            print("⚠️ 无有效任务生成。")
            return

        # 缓冲区配置
        fact_buffer: List[AtomicMemoryUnit] = []
        EMBED_FLUSH_THRESHOLD = 128

        total_extracted = 0

        # 追踪后台任务的集合，防止任务被垃圾回收，并用于最终等待
        background_storage_tasks = set()

        print(f"  > 正在并行执行 {len(llm_tasks)} 个 LLM 提取任务...")

        # 3. 流式处理
        for finished_task in asyncio.as_completed(llm_tasks):
            try:
                extracted_facts: List[AtomicMemoryUnit] = await finished_task

                if extracted_facts:
                    fact_buffer.extend(extracted_facts)
                    total_extracted += len(extracted_facts)

                    if len(fact_buffer) >= EMBED_FLUSH_THRESHOLD:
                        # 必须切片复制一份 list，因为 fact_buffer 马上要清空
                        chunk_to_process = fact_buffer[:]
                        fact_buffer = []

                        # 使用 create_task 将存储操作放入后台运行 (Fire-and-forget)
                        # 这样主循环可以立即继续处理下一个 LLM 结果，而不需要等待 Embedding/DB 完成
                        task = asyncio.create_task(self._process_facts_chunk(chunk_to_process, source_suffix))

                        # 注册任务到集合中
                        background_storage_tasks.add(task)
                        # 任务完成后自动从集合中移除，保持集合精简
                        task.add_done_callback(background_storage_tasks.discard)

            except Exception as e:
                print(f"❌ LLM Task Exception: {e}")

        # 5. 处理剩余尾部数据
        if fact_buffer:
            # 同样作为后台任务提交
            task = asyncio.create_task(self._process_facts_chunk(fact_buffer, source_suffix))
            background_storage_tasks.add(task)
            task.add_done_callback(background_storage_tasks.discard)

        # 确保所有后台存储任务完成
        # 虽然是"异步"，但在整个函数退出前，必须确保数据都落盘了，否则程序结束会杀掉未完成的任务
        if background_storage_tasks:

            await asyncio.gather(*background_storage_tasks)

        print(f"> 处理完成。共提取并存储 {total_extracted} 个原子事实。")

    async def _process_facts_chunk(self, facts: List[AtomicMemoryUnit], source_suffix: str):
        """后台 Embedding 处理"""
        if not facts: return

        fact_texts = [amu.fact_text for amu in facts]
        loop = asyncio.get_running_loop()

        # A. VQ-VAE (CPU/GPU)
        if self.vq_model:
            try:
                codes_list = await loop.run_in_executor(None, self.vq_model.get_indices, fact_texts)
                for amu, codes in zip(facts, codes_list):
                    amu.code_indices = codes
            except Exception as e:
                print(f"❌ VQ-VAE Chunk Error: {e}")

        # B. Embedding (CPU/GPU)
        try:
            embeddings = await loop.run_in_executor(None, self.embedder.embed_batch, fact_texts)
            for amu, vec in zip(facts, embeddings):
                amu.embedding = vec
        except Exception as e:
            print(f"❌ Embedding Chunk Error: {e}")

        # C. Storage (IO)
        try:
            for amu in facts:
                self.meta_db.add_atomic_memory(amu)

            # 写入 DB 后立即更新内存中的全局统计信息
            # 这步操作必须在 Code 已经生成之后，且 facts 已经有完整数据时进行
            self.meta_db.update_stats(facts)

            vector_data = []
            for amu in facts:
                if amu.embedding:
                    vector_data.append((amu.id, amu.embedding, amu.fact_text))

            if vector_data:
                self.vec_db.add_vectors_batch("atomic_memory", vector_data)

            if self.trace_log_dir:
                self.export_extraction_trace(facts, suffix=source_suffix)

        except Exception as e:
            print(f"❌ Storage Chunk Error: {e}")

    def shutdown_databases(self):
        print("\n--- Closing Databases ---")
        self.vec_db.close()
        self.meta_db.close()
