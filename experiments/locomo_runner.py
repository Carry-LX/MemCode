

import os
import json
import shutil
import asyncio
import time
from typing import List, Dict, Any, Union
from datetime import datetime
import nltk

# 导入项目模块
from src.core.metadata_db_handler import MetadataDBHandler
from src.core.vector_db_handler import VectorDBHandler
from src.processing.text_splitter import ConversationSplitter

from src.utils.embedding_bge import EmbeddingModel
from src.utils.vqvae_inference_v10 import ModelInference

from src.processing.locomo_memory_pipeline import AsyncProcessingPipeline
from release_runtime import selected_query_engine
QueryEngine = selected_query_engine()
from src.utils.llm_interface import OpenAIInterface
from src.utils.llm_judge import LLMJudge

# 引入演化模块
from src.processing.locomo_memory_evolver import MemoryConsolidator

# Settings supplied by the selected run profile.
from settings import active_config
CONFIG = active_config()
DATA_PATH = str(CONFIG.data_path)
DATASET_TYPE = CONFIG.dataset
MODEL_PATH = CONFIG.embedding_model
Model_Dimension = 768
SKIP_LOCOMO_ADVERSARIAL = True
TEST_ROOT_DIR = str(CONFIG.output_dir)
OPENAI_API_KEY = CONFIG.api_key
BaseUrl = CONFIG.base_url
MEMORY_LLM_MODEL = CONFIG.memory_model
QA_LLM_MODEL = CONFIG.qa_model
JUDGE_LLM_MODEL = CONFIG.judge_model
ENABLE_CONSOLIDATION = CONFIG.consolidation
SOURCE_ROOT_DIR = str(CONFIG.source_dir) if CONFIG.source_dir else ""


def map_question_type_to_judge_category(question_type: str) -> str:
    """LongMemEval 的原始映射逻辑"""
    type_mapping = {
        "single-session": LLMJudge.STANDARD_TASK,
        "single-session-user": LLMJudge.STANDARD_TASK,
        "single-session-assistant": LLMJudge.STANDARD_TASK,
        "multi-session": LLMJudge.STANDARD_TASK,
        "temporal-reasoning": LLMJudge.TEMPORAL_REASONING,
        "knowledge-update": LLMJudge.KNOWLEDGE_UPDATE,
        "single-session-preference": LLMJudge.PREFERENCE_TASK,
        "abstention": LLMJudge.ABSTENTION_TASK,
    }
    return type_mapping.get(question_type, LLMJudge.STANDARD_TASK)


def map_locomo_type_to_judge_category(question_type: Union[int, str]) -> str:
    try:
        cat_id = int(question_type)
    except (ValueError, TypeError):
        q_type_str = str(question_type).lower().strip()
        if "temporal" in q_type_str: return LLMJudge.TEMPORAL_REASONING
        if "adversarial" in q_type_str: return LLMJudge.ABSTENTION_TASK
        return LLMJudge.STANDARD_TASK

    if cat_id == 2:
        return LLMJudge.TEMPORAL_REASONING
    elif cat_id == 5:
        return LLMJudge.ABSTENTION_TASK
    else:
        return LLMJudge.STANDARD_TASK


def load_longmemeval_dataset(data_path: str) -> List[Dict[str, Any]]:
    print(f"正在加载 LongMemEval 数据集: {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"加载了 {len(data)} 个测试项。")
        for item in data:
            item['dataset_type'] = 'longmemeval'
        return data[:CONFIG.max_rows] if CONFIG.max_rows is not None else data
    except FileNotFoundError:
        print(f"错误: 找不到文件 {data_path}")
        return []


def load_locomo_dataset(data_path: str) -> List[Dict[str, Any]]:
    print(f"正在加载 LoCoMo 数据集: {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {data_path}")
        return []

    processed_rows = []
    for row_idx, entry in enumerate(raw_data):
        conversation = entry.get('conversation', {})
        qa_list = entry.get('qa', [])

        processed_rows.append({
            "dataset_type": "locomo",
            "row_id": f"locomo_row_{row_idx}",
            "conversation": conversation,
            "qa_list": qa_list
        })

    print(f"数据集加载完成，共解析出 {len(processed_rows)} 个对话场景 (Row)。")
    return processed_rows[:CONFIG.max_rows] if CONFIG.max_rows is not None else processed_rows


def init_components_once():
    """初始化静态组件 (只运行一次)"""
    print("正在初始化核心模型组件 (Static)...")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', raise_on_error=True)

    splitter = ConversationSplitter()
    embedder = EmbeddingModel(model_path=MODEL_PATH, device=CONFIG.device)
    llm_interface = OpenAIInterface(api_key=OPENAI_API_KEY, model=MEMORY_LLM_MODEL, base_url=BaseUrl)
    judge_llm_interface = OpenAIInterface(api_key=OPENAI_API_KEY, model=JUDGE_LLM_MODEL, base_url=BaseUrl)

    # [Monkey Patch] 动态植入计数器
    if not hasattr(llm_interface, 'call_count'):
        llm_interface.call_count = 0
        original_completion = llm_interface.get_completion

        def wrapped_completion(*args, **kwargs):
            llm_interface.call_count += 1
            return original_completion(*args, **kwargs)

        llm_interface.get_completion = wrapped_completion
        print(">> [System] 已启用 LLM 调用计数追踪 (Monkey Patch Active)")

    judge = LLMJudge(llm_interface=judge_llm_interface)
    print("正在初始化 VQ-VAE (Static)...")
    vq_model_static = ModelInference(device_str=CONFIG.device)
    return splitter, embedder, llm_interface, judge_llm_interface, judge, vq_model_static


def init_databases_and_pipeline(splitter, embedder, llm_interface, specific_storage_dir, vq_model_instance):
    meta_path = os.path.join(specific_storage_dir, "metadata.db")
    vec_path = os.path.join(specific_storage_dir, "vector_stores")
    os.makedirs(vec_path, exist_ok=True)

    meta_db = MetadataDBHandler(db_path=meta_path)
    vec_db = VectorDBHandler(storage_path=vec_path)

    vec_db.create_or_load_index("atomic_memory", Model_Dimension)

    pipeline = AsyncProcessingPipeline(splitter, embedder, llm_interface, meta_db, vec_db, vq_model_instance)
    query_engine = QueryEngine(embedder, meta_db, vec_db, llm_interface, vq_model_instance)

    return pipeline, query_engine, meta_db, vec_db


def save_qa_trace_log(trace_data: Dict[str, Any], log_dir: str, q_id: str):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_path = os.path.join(log_dir, f"qa_trace_{q_id}.json")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(trace_data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Warning: Failed to save trace log for {q_id}: {e}")


async def run_qa_evaluation(query_engine, judge, dataset_type, question, correct_answer, question_id, fact_num,
                            question_type="standard", log_dir=None, query_date=None):
    """QA 执行逻辑"""
    qa_start_time = time.time()

    result_entry = {
        "question_id": question_id,
        "question": question,
        "ground_truth": correct_answer,
        "generated_answer": "",
        "is_correct": False,
        "judge_category": "",
        "raw_category": question_type,
        "error": None
    }

    perf_stats = {
        "latency_seconds": 0.0,
        "generation_tokens": 0,
        "generation_prompt_tokens": 0,
        "generation_completion_tokens": 0
    }

    full_trace_record = {
        "question_id": question_id,
        "timestamp": datetime.now().isoformat(),
        "dataset_type": dataset_type,
        "question": question,
        "question_type_raw": question_type,
        "query_reference_time": query_date,
        "retrieval_trace": {},
        "generation": {},
        "evaluation": {}
    }

    try:
        query_text = question
        if dataset_type != "locomo":
            query_text = query_text.replace(" I ", " the user ").replace(" my ", " the user's ")

        # [Step 1] Query
        query_result = query_engine.query(
            query_text,
            total_atomic_count=fact_num,
            k=20,
            query_date=query_date
        )

        final_prompt = query_result["final_prompt"]
        trace_info = query_result["trace_info"]
        full_trace_record["retrieval_trace"] = trace_info

        # [Step 2] Generate (独立实例)
        llm_qa = OpenAIInterface(api_key=OPENAI_API_KEY, model=QA_LLM_MODEL, base_url=BaseUrl)
        generated_answer = await asyncio.to_thread(llm_qa.get_completion, final_prompt)

        # [Stats] QA 消耗
        qa_end_time = time.time()
        perf_stats["latency_seconds"] = qa_end_time - qa_start_time
        usage_qa = llm_qa.get_total_usage() or {}
        perf_stats["generation_tokens"] = usage_qa.get("total_tokens", 0)
        perf_stats["generation_prompt_tokens"] = usage_qa.get("total_prompt_tokens", 0)
        perf_stats["generation_completion_tokens"] = usage_qa.get("total_completion_tokens", 0)

        result_entry["generated_answer"] = generated_answer

        full_trace_record["generation"] = {
            "final_prompt_used": final_prompt,
            "model_answer": generated_answer,
            "ground_truth": correct_answer
        }

        # [Step 3] Judge
        if dataset_type == "locomo":
            judge_category = map_locomo_type_to_judge_category(question_type)
        else:
            judge_category = map_question_type_to_judge_category(question_type)

        result_entry["judge_category"] = judge_category

        def run_judge_logic():
            if judge_category == LLMJudge.ABSTENTION_TASK or 'abs' in str(question_id):
                res = judge.evaluate(judge_category, question, generated_answer, explanation=correct_answer,
                                     query_date=query_date)
            elif judge_category == LLMJudge.PREFERENCE_TASK:
                res = judge.evaluate(judge_category, question, generated_answer, rubric=correct_answer,
                                     query_date=query_date)
            else:
                res = judge.evaluate(judge_category, question, generated_answer, correct_answer=correct_answer,
                                     query_date=query_date)

            return res

        judge_result = await asyncio.to_thread(run_judge_logic)
        is_correct = judge_result

        result_entry["is_correct"] = is_correct

        full_trace_record["evaluation"] = {
            "is_correct": is_correct,
            "category": judge_category,
            "judge_explanation_used": correct_answer,
            "judge_reference_time_used": query_date
        }

        if log_dir:
            save_qa_trace_log(full_trace_record, log_dir, question_id)

    except Exception as e:
        result_entry["error"] = str(e)
        print(f"QA Error ({question_id}): {e}")

    result_entry["perf_stats"] = perf_stats
    return result_entry


async def main_async():
    if DATASET_TYPE == "locomo":
        test_data = load_locomo_dataset(DATA_PATH)
    else:
        test_data = load_longmemeval_dataset(DATA_PATH)

    if not test_data:
        print("未加载到数据，退出。")
        return

    if os.path.exists(TEST_ROOT_DIR) and os.listdir(TEST_ROOT_DIR):
        raise FileExistsError(f"Choose a new RUN_LABEL; results already exist: {TEST_ROOT_DIR}")
    os.makedirs(TEST_ROOT_DIR, exist_ok=True)

    from release_runtime import write_run_manifest
    write_run_manifest(CONFIG)

    # 全局统计累加器 - 分三类
    # Memory construction accumulators (prompt/completion/total and calls)
    accum_memory_prompt_tokens = 0
    accum_memory_completion_tokens = 0
    accum_memory_tokens = 0
    accum_memory_calls = 0

    # Evolution accumulators
    accum_evolution_prompt_tokens = 0
    accum_evolution_completion_tokens = 0
    accum_evolution_tokens = 0
    accum_evolution_calls = 0

    # QA generation accumulators
    accum_qa_prompt_tokens = 0
    accum_qa_completion_tokens = 0
    accum_qa_tokens = 0

    all_memory_times = []  # 初始构建时间
    all_evolution_times = []  # 演化时间
    all_qa_times = []  # 问答时间

    # 初始化组件
    splitter, embedder, llm_interface, judge_llm_interface, judge, vq_model_static = init_components_once()

    results = []
    print(f"\n=== 开始测试 (数据结果保留在 {TEST_ROOT_DIR} 下的各个子目录中) ===")

    for idx, item in enumerate(test_data):
        print(f"\nProcessing Row {idx + 1}/{len(test_data)} ...")

        unique_folder_name = f"row_{idx}_data"
        current_row_storage_path = os.path.join(TEST_ROOT_DIR, unique_folder_name)
        os.makedirs(current_row_storage_path, exist_ok=True)

        pipeline, query_engine, meta_db, vec_db = init_databases_and_pipeline(
            splitter, embedder, llm_interface,
            specific_storage_dir=current_row_storage_path,
            vq_model_instance=vq_model_static
        )

        try:
            # ==========================================
            # 阶段 1: 记忆构建 (Initial Memory Construction)
            # ==========================================
            t_mem_start = time.time()
            usage_before_mem = llm_interface.get_total_usage() or {}
            tokens_before_mem = usage_before_mem.get('total_tokens', 0)
            prompts_before_mem = usage_before_mem.get('total_prompt_tokens', 0)
            comps_before_mem = usage_before_mem.get('total_completion_tokens', 0)
            calls_before_mem = getattr(llm_interface, 'call_count', 0)

            if item["dataset_type"] == "locomo":
                source_id = item["row_id"]
                await pipeline.process_locomo_data_async(
                    item["conversation"], source_id, log_dir=current_row_storage_path
                )
            else:
                sessions = item["haystack_sessions"]
                timestamps = item["haystack_dates"]
                q_id = item["question_id"]
                print(f"  > LongMemEval: {len(sessions)} sessions.")
                await pipeline.process_longmemeval_data_async(
                    sessions, timestamps, q_id, log_dir=current_row_storage_path
                )

            # [统计] 记忆构建
            t_mem_end = time.time()
            usage_after_mem = llm_interface.get_total_usage() or {}
            tokens_after_mem = usage_after_mem.get('total_tokens', 0)
            prompts_after_mem = usage_after_mem.get('total_prompt_tokens', 0)
            comps_after_mem = usage_after_mem.get('total_completion_tokens', 0)
            calls_after_mem = getattr(llm_interface, 'call_count', 0)

            mem_duration = t_mem_end - t_mem_start
            mem_total_delta = tokens_after_mem - tokens_before_mem
            mem_prompt_delta = prompts_after_mem - prompts_before_mem
            mem_completion_delta = comps_after_mem - comps_before_mem
            mem_calls_delta = calls_after_mem - calls_before_mem

            accum_memory_tokens += mem_total_delta
            accum_memory_prompt_tokens += mem_prompt_delta
            accum_memory_completion_tokens += mem_completion_delta
            accum_memory_calls += mem_calls_delta
            all_memory_times.append(mem_duration)

            count_fact = meta_db.get_total_atomic_count()
            print(f"  > [Build] Facts: {count_fact} | Time: {mem_duration:.2f}s | Prompt: {mem_prompt_delta} | Completion: {mem_completion_delta} | Total: {mem_total_delta}")

            # ==========================================
            # 阶段 2: 记忆演化 (Memory Evolution / Optimization)
            # ==========================================
            # [计时与统计 - 演化]
            t_evo_start = time.time()
            usage_before_evo = llm_interface.get_total_usage() or {}
            tokens_before_evo = usage_before_evo.get('total_tokens', 0)
            prompts_before_evo = usage_before_evo.get('total_prompt_tokens', 0)
            comps_before_evo = usage_before_evo.get('total_completion_tokens', 0)
            calls_before_evo = getattr(llm_interface, 'call_count', 0)

            if ENABLE_CONSOLIDATION:
                consolidator = MemoryConsolidator(
                    meta_db=meta_db, vec_db=vec_db, fact_num=count_fact,
                    embedder=embedder, llm=llm_interface, vq_model=vq_model_static,
                    max_concurrency=10,
                )
                await consolidator.execute_evolution_async()

            # [统计] 演化结束
            t_evo_end = time.time()
            usage_after_evo = llm_interface.get_total_usage() or {}
            tokens_after_evo = usage_after_evo.get('total_tokens', 0)
            prompts_after_evo = usage_after_evo.get('total_prompt_tokens', 0)
            comps_after_evo = usage_after_evo.get('total_completion_tokens', 0)
            calls_after_evo = getattr(llm_interface, 'call_count', 0)

            evo_duration = t_evo_end - t_evo_start
            evo_total_delta = tokens_after_evo - tokens_before_evo
            evo_prompt_delta = prompts_after_evo - prompts_before_evo
            evo_completion_delta = comps_after_evo - comps_before_evo
            evo_calls_delta = calls_after_evo - calls_before_evo

            accum_evolution_tokens += evo_total_delta
            accum_evolution_prompt_tokens += evo_prompt_delta
            accum_evolution_completion_tokens += evo_completion_delta
            accum_evolution_calls += evo_calls_delta
            all_evolution_times.append(evo_duration)

            # 再次统计数量 (查看演化后的变化)
            count_fact_final = meta_db.get_total_atomic_count()
            count_fact = count_fact_final  # 更新用于 QA 的数量
            print(
                f"  > [Evo] Facts Post-Evolution: {count_fact_final} | Time: {evo_duration:.2f}s | Prompt: {evo_prompt_delta} | Completion: {evo_completion_delta} | Total: {evo_total_delta}")

            # ==========================================
            # 阶段 3: 问答 (QA)
            # ==========================================
            if item["dataset_type"] == "locomo":
                print(f"\n  >>> 启动并发 QA ...")
                CONCURRENCY_LIMIT = 15
                qa_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

                async def process_single_qa(q_idx, qa_pair):
                    async with qa_semaphore:
                        q_id = f"{source_id}_q{q_idx}"
                        q_category = qa_pair.get('category', 1)
                        if SKIP_LOCOMO_ADVERSARIAL and q_category == 5: return None

                        ground_truth = str(qa_pair.get('answer'))
                        if q_category == 5 and not ground_truth:
                            ground_truth = str(qa_pair.get('adversarial_answer', 'Unknown'))

                        res = await run_qa_evaluation(
                            query_engine, judge, "locomo",
                            qa_pair['question'], ground_truth, q_id, count_fact,
                            question_type=q_category, log_dir=current_row_storage_path
                        )
                        return res

                tasks = [process_single_qa(i, qa) for i, qa in enumerate(item["qa_list"])]
                batch_results = await asyncio.gather(*tasks)
                valid_results = [r for r in batch_results if r is not None]
                results.extend(valid_results)

            else:
                print(f"  > Querying: {item['question']}")
                res = await run_qa_evaluation(
                    query_engine, judge, "longmemeval",
                    item['question'], item['answer'], item["question_id"], count_fact,
                    item.get('question_type', 'standard'),
                    log_dir=current_row_storage_path,
                    query_date=item.get("question_date")
                )
                valid_results = [res]
                results.append(res)

            # [统计] QA 消耗
            for res in valid_results:
                stats = res.get("perf_stats", {})
                q_time = stats.get("latency_seconds", 0)
                q_gen_tokens = stats.get("generation_tokens", 0)
                q_gen_prompt = stats.get("generation_prompt_tokens", 0)
                q_gen_completion = stats.get("generation_completion_tokens", 0)

                all_qa_times.append(q_time)
                accum_qa_tokens += q_gen_tokens
                accum_qa_prompt_tokens += q_gen_prompt
                accum_qa_completion_tokens += q_gen_completion

        finally:
            pipeline.shutdown_databases()

    # 最终统计输出
    print("\n" + "=" * 60)
    print("详细统计报告 (Detailed Statistics)")
    print("=" * 60)

    # 4.1 按类别统计
    stats_by_category = {}
    locomo_cat_names = {
        1: "Standard (Multi-Hop)",
        2: "Temporal Reasoning",
        3: "Standard (Open Domain)",
        4: "Standard (Single-Hop/Basic)",
        5: "Adversarial/Abstention"
    }

    for r in results:
        raw_cat = r.get("raw_category", "Unknown")
        if raw_cat not in stats_by_category: stats_by_category[raw_cat] = {"total": 0, "correct": 0}
        stats_by_category[raw_cat]["total"] += 1
        if r["is_correct"]: stats_by_category[raw_cat]["correct"] += 1

    for cat in sorted(stats_by_category.keys(), key=lambda x: str(x)):
        s = stats_by_category[cat]
        acc = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0.0

        display_name = f"Category {cat}"
        if DATASET_TYPE == "locomo":
            try:
                cat_int = int(cat)
                if cat_int in locomo_cat_names:
                    display_name = f"Category {cat_int} [{locomo_cat_names[cat_int]}]"
            except:
                pass

        print(f"类别: {display_name:<38} | 总数: {s['total']:<4} | 正确: {s['correct']:<4} | 准确率: {acc:.2f}%")

    # 4.3 总体 Token 和 性能统计 (三阶段分离)
    print("-" * 60)

    # Judge 使用全局独立实例；直接读取其累计 token，避免并发差值统计不准
    judge_usage = judge_llm_interface.get_total_usage() or {}
    accum_judge_prompt_tokens = judge_usage.get('total_prompt_tokens', 0)
    accum_judge_completion_tokens = judge_usage.get('total_completion_tokens', 0)
    accum_judge_tokens = judge_usage.get('total_tokens', 0)

    # 真实总计 = Memory + Evolution + Judge + QA_Gen
    grand_total_tokens = accum_memory_tokens + accum_evolution_tokens + accum_judge_tokens + accum_qa_tokens
    grand_total_prompt_tokens = (
        accum_memory_prompt_tokens
        + accum_evolution_prompt_tokens
        + accum_qa_prompt_tokens
        + accum_judge_prompt_tokens
    )
    grand_total_completion_tokens = (
        accum_memory_completion_tokens
        + accum_evolution_completion_tokens
        + accum_qa_completion_tokens
        + accum_judge_completion_tokens
    )

    total_q = len(results)
    total_correct = sum(1 for r in results if r["is_correct"])
    total_acc = (total_correct / total_q * 100) if total_q else 0

    avg_mem_time = sum(all_memory_times) / len(all_memory_times) if all_memory_times else 0
    avg_evo_time = sum(all_evolution_times) / len(all_evolution_times) if all_evolution_times else 0
    avg_qa_time = sum(all_qa_times) / len(all_qa_times) if all_qa_times else 0

    print(f"【总体表现】")
    print(f"总计问题数: {total_q}")
    print(f"总计正确数: {total_correct}")
    print(f"总体准确率: {total_acc:.2f}%")
    print("-" * 30)
    print(f"【资源消耗详情 (Tokens)】")
    print(f"1. 记忆构建:   Prompt {accum_memory_prompt_tokens:,} | Completion {accum_memory_completion_tokens:,} | Total {accum_memory_tokens:,} (调用: {accum_memory_calls})")
    print(f"2. 记忆演化:   Prompt {accum_evolution_prompt_tokens:,} | Completion {accum_evolution_completion_tokens:,} | Total {accum_evolution_tokens:,} (调用: {accum_evolution_calls}) <-- 优化消耗")
    print(f"3. 问答生成:   Prompt {accum_qa_prompt_tokens:,} | Completion {accum_qa_completion_tokens:,} | Total {accum_qa_tokens:,}")
    print(f"4. 自动评判:   Prompt {accum_judge_prompt_tokens:,} | Completion {accum_judge_completion_tokens:,} | Total {accum_judge_tokens:,}")
    print(f"----------------------")
    print(f"Token 总计:    {grand_total_tokens:,}")
    print("-" * 30)
    print(f"【Token 输入/输出分解 (Prompt + Completion)】")
    print(f"输入 (Prompt)  总计: {grand_total_prompt_tokens:,}")
    print(f"输出 (Completion) 总计: {grand_total_completion_tokens:,}")
    print("-" * 30)
    print(f"【平均时耗 (Seconds)】")
    print(f"记忆构建: {avg_mem_time:.2f} s/Session")
    print(f"记忆演化: {avg_evo_time:.2f} s/Session <-- 优化耗时")
    print(f"问答响应: {avg_qa_time:.2f} s/Question")
    print("=" * 60)

    # 5. 保存结果
    output_file = os.path.join(TEST_ROOT_DIR, f"eval_results_{DATASET_TYPE}_persist.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"详细结果Json已保存到: {output_file}")

    # 保存性能日志
    perf_log = {
        "memory": {
            "prompt": accum_memory_prompt_tokens,
            "completion": accum_memory_completion_tokens,
            "total": accum_memory_tokens,
            "calls": accum_memory_calls
        },
        "evolution": {
            "prompt": accum_evolution_prompt_tokens,
            "completion": accum_evolution_completion_tokens,
            "total": accum_evolution_tokens,
            "calls": accum_evolution_calls
        },
        "qa_generation": {
            "prompt": accum_qa_prompt_tokens,
            "completion": accum_qa_completion_tokens,
            "total": accum_qa_tokens
        },
        "judge": {
            "prompt": accum_judge_prompt_tokens,
            "completion": accum_judge_completion_tokens,
            "total": accum_judge_tokens
        },
        "total_tokens": grand_total_tokens,
        "prompt_tokens_total": grand_total_prompt_tokens,
        "completion_tokens_total": grand_total_completion_tokens,
        "avg_mem_time": avg_mem_time,
        "avg_evo_time": avg_evo_time,
        "avg_qa_time": avg_qa_time
    }
    with open(os.path.join(TEST_ROOT_DIR, f"eval_performance_{DATASET_TYPE}.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(perf_log, ensure_ascii=False, indent=2))

    print(f"中间数据库及详细Trace日志已保留在目录: {TEST_ROOT_DIR}")
    # 生成 JSON 格式的准确率汇总
    try:
        os.makedirs(TEST_ROOT_DIR, exist_ok=True)
        summary = {}

        # overall 已在上面计算过，但为了稳健性在此重新计算
        total_q = len(results)
        total_correct = sum(1 for r in results if r.get("is_correct"))
        total_acc = (total_correct / total_q * 100) if total_q else 0.0
        summary["overall"] = {
            "total_questions": total_q,
            "total_correct": total_correct,
            "accuracy_percent": round(total_acc, 4)
        }

        # by_raw_category: 使用之前构建的 stats_by_category（若不存在则重建）
        try:
            stats = stats_by_category
        except NameError:
            stats = {}
            for r in results:
                raw_cat = r.get("raw_category", "Unknown")
                if raw_cat not in stats:
                    stats[raw_cat] = {"total": 0, "correct": 0}
                stats[raw_cat]["total"] += 1
                if r.get("is_correct"):
                    stats[raw_cat]["correct"] += 1

        by_cat = {}
        for cat, v in stats.items():
            tot = v.get("total", 0)
            cor = v.get("correct", 0)
            acc = (cor / tot * 100) if tot else 0.0
            entry = {
                "total": tot,
                "correct": cor,
                "accuracy_percent": round(acc, 4)
            }
            # 如果是 LoCoMo 数据集，尽可能添加人类可读的类别名称
            if DATASET_TYPE == "locomo":
                try:
                    cat_int = int(cat)
                    if cat_int in locomo_cat_names:
                        entry["display_name"] = locomo_cat_names[cat_int]
                except Exception:
                    # 如果无法转换则忽略
                    pass

            by_cat[str(cat)] = entry

        summary["by_raw_category"] = by_cat

        summary_path = os.path.join(TEST_ROOT_DIR, "accuracy_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as sf:
            sf.write(json.dumps(summary, ensure_ascii=False, indent=2))

        print(f"准确率汇总已保存到: {summary_path}")
    except Exception as e:
        print(f"Warning: 生成准确率汇总时出错: {e}")


if __name__ == '__main__':
    asyncio.run(main_async())
