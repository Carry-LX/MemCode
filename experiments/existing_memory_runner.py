

import os
import json
import shutil
import asyncio
from typing import List, Dict, Any, Union
from datetime import datetime
import nltk
import time

# 导入项目模块
from src.core.metadata_db_handler import MetadataDBHandler
from src.core.vector_db_handler import VectorDBHandler
from src.processing.text_splitter import ConversationSplitter

from src.utils.embedding_bge import EmbeddingModel
from src.utils.vqvae_inference_v10 import ModelInference # 确保引用正确

from src.processing.longmemeval_memory_pipeline import AsyncProcessingPipeline
from release_runtime import selected_query_engine
QueryEngine = selected_query_engine()
from src.utils.llm_interface import OpenAIInterface
from src.utils.llm_judge import LLMJudge


from src.processing.longmemeval_memory_evolver import MemoryConsolidator


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
    """LoCoMo 数据集的映射逻辑"""
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
    """加载 LongMemEval 数据集"""
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
    """加载 LoCoMo 数据集 (按行返回)"""
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

    if not hasattr(llm_interface, 'call_count'):
        llm_interface.call_count = 0
        original_completion = llm_interface.get_completion

        def wrapped_completion(*args, **kwargs):
            llm_interface.call_count += 1
            return original_completion(*args, **kwargs)

        llm_interface.get_completion = wrapped_completion

    judge_llm_interface = OpenAIInterface(api_key=OPENAI_API_KEY, model=JUDGE_LLM_MODEL, base_url=BaseUrl)
    judge = LLMJudge(llm_interface=judge_llm_interface)

    print("正在初始化 VQ-VAE (Static)...")
    vq_model_static = ModelInference(device_str=CONFIG.device)
    return splitter, embedder, llm_interface, judge_llm_interface, judge, vq_model_static


def has_non_ascii_path(path: str) -> bool:
    try:
        path.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def extract_row_index(folder_name: str) -> int:
    try:
        parts = folder_name.split('_')
        if len(parts) >= 2 and parts[0] == 'row':
            return int(parts[1])
    except (ValueError, TypeError):
        pass
    return -1


def get_expected_trace_count(item: Dict[str, Any]) -> int:
    if item.get("dataset_type") == "locomo":
        qa_list = item.get("qa_list", [])
        return sum(
            1 for qa in qa_list
            if not (SKIP_LOCOMO_ADVERSARIAL and qa.get('category', 1) == 5)
        )
    return 1


def is_row_completed(row_dir: str, item: Dict[str, Any]) -> bool:
    if not os.path.isdir(row_dir):
        return False

    expected_count = get_expected_trace_count(item)
    qa_trace_files = [
        name for name in os.listdir(row_dir)
        if name.startswith("qa_trace_") and name.endswith(".json")
    ]
    return len(qa_trace_files) >= expected_count


def find_resume_start_index(test_data: List[Dict[str, Any]]) -> int:
    for idx, item in enumerate(test_data):
        row_dir = os.path.join(TEST_ROOT_DIR, f"row_{idx}_data")
        if not is_row_completed(row_dir, item):
            return max(0, idx - 1)
    return len(test_data)


def cleanup_rows_from_index(start_idx: int):
    if not os.path.isdir(TEST_ROOT_DIR):
        return

    for entry in os.listdir(TEST_ROOT_DIR):
        row_idx = extract_row_index(entry)
        if row_idx >= start_idx:
            row_path = os.path.join(TEST_ROOT_DIR, entry)
            if os.path.isdir(row_path):
                from release_runtime import safe_remove_run_directory
                safe_remove_run_directory(row_path, TEST_ROOT_DIR)


def _build_result_entry_from_trace(trace_obj: Dict[str, Any], existing_result: Dict[str, Any] = None) -> Dict[str, Any]:
    generation = trace_obj.get("generation", {})
    evaluation = trace_obj.get("evaluation", {})
    result_entry = {
        "question_id": trace_obj.get("question_id"),
        "question": trace_obj.get("question"),
        "ground_truth": generation.get("ground_truth", ""),
        "generated_answer": generation.get("model_answer", ""),
        "is_correct": evaluation.get("is_correct", False),
        "judge_category": evaluation.get("category", ""),
        "raw_category": trace_obj.get("question_type_raw", "Unknown"),
        "error": None,
        "perf_stats": {
            "latency_seconds": 0.0,
            "generation_tokens": 0,
            "generation_prompt_tokens": 0,
            "generation_completion_tokens": 0
        }
    }

    if existing_result:
        result_entry["perf_stats"] = existing_result.get("perf_stats", result_entry["perf_stats"])
        result_entry["error"] = existing_result.get("error")
        result_entry["judge_category"] = existing_result.get("judge_category", result_entry["judge_category"])
        result_entry["raw_category"] = existing_result.get("raw_category", result_entry["raw_category"])

    return result_entry


def load_existing_results(test_data: List[Dict[str, Any]], stop_before_idx: int, dataset_type: str) -> List[Dict[str, Any]]:
    existing_results_path = os.path.join(TEST_ROOT_DIR, f"eval_results_{dataset_type}_skipped.json")
    existing_result_map = {}

    if os.path.exists(existing_results_path):
        try:
            with open(existing_results_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            existing_result_map = {
                item.get("question_id"): item
                for item in existing_results
                if item.get("question_id")
            }
        except Exception as e:
            print(f"  [Resume] 读取旧结果汇总失败，将仅从 qa_trace 重建: {e}")

    loaded_results = []
    for idx in range(stop_before_idx):
        row_dir = os.path.join(TEST_ROOT_DIR, f"row_{idx}_data")
        if not os.path.isdir(row_dir):
            continue

        trace_paths = sorted(
            [
                os.path.join(row_dir, name)
                for name in os.listdir(row_dir)
                if name.startswith("qa_trace_") and name.endswith(".json")
            ]
        )

        for trace_path in trace_paths:
            try:
                with open(trace_path, 'r', encoding='utf-8') as f:
                    trace_obj = json.load(f)
                question_id = trace_obj.get("question_id")
                loaded_results.append(
                    _build_result_entry_from_trace(trace_obj, existing_result_map.get(question_id))
                )
            except Exception as e:
                print(f"  [Resume] 跳过损坏的 trace 文件 {trace_path}: {e}")

    return loaded_results


def copy_existing_databases(src_row_path: str, dst_row_path: str):
    """Copy a complete memory store into a separate output directory.

    Copy metadata, codebook statistics, vector index and ID map together."""
    if not os.path.exists(src_row_path):
        raise FileNotFoundError(f"源数据目录不存在: {src_row_path}。请检查 SOURCE_ROOT_DIR 配置是否正确。")

    required = ("metadata.db", "metadata.db.stats.json", "vector_stores/atomic_memory.faiss",
                "vector_stores/atomic_memory_map.json")
    for relative in required:
        if not os.path.isfile(os.path.join(src_row_path, relative)):
            raise FileNotFoundError(f"Incomplete source memory: {src_row_path}/{relative}")

    # 1. 复制 metadata.db
    src_meta = os.path.join(src_row_path, "metadata.db")
    dst_meta = os.path.join(dst_row_path, "metadata.db")
    if os.path.exists(src_meta):
        shutil.copy2(src_meta, dst_meta)
        print(f"  [Restore] Copied metadata.db from {src_row_path}")
    else:
        print(f"  [Warning] Source metadata.db not found in {src_row_path}")

    # 1.1  复制 metadata.db.stats.json (统计文件)
    src_stats = os.path.join(src_row_path, "metadata.db.stats.json")
    dst_stats = os.path.join(dst_row_path, "metadata.db.stats.json")

    if os.path.exists(src_stats):
        shutil.copy2(src_stats, dst_stats)
        print(f"  [Restore] Copied metadata.db.stats.json from {src_row_path}")
    else:
        # Source validation requires the codebook statistics file.
        print(f"  [Warning] Source metadata.db.stats.json not found in {src_row_path}")

    # 2. 复制 vector_stores 文件夹
    src_vec = os.path.join(src_row_path, "vector_stores")
    dst_vec = os.path.join(dst_row_path, "vector_stores")
    if os.path.exists(src_vec):
        if os.path.exists(dst_vec):
            from release_runtime import safe_remove_run_directory
            safe_remove_run_directory(dst_vec, TEST_ROOT_DIR)  # 如果目标存在先删除，保证干净复制
        shutil.copytree(src_vec, dst_vec)
        print(f"  [Restore] Copied vector_stores/ from {src_row_path}")
        copied_faiss = os.path.join(dst_vec, "atomic_memory.faiss")
        copied_map = os.path.join(dst_vec, "atomic_memory_map.json")
        print(
            f"  [Restore Check] atomic_memory.faiss exists={os.path.exists(copied_faiss)} | "
            f"atomic_memory_map.json exists={os.path.exists(copied_map)}"
        )
    else:
        print(f"  [Warning] Source vector_stores/ not found in {src_row_path}")


def init_databases_and_pipeline(splitter, embedder, llm_interface, specific_storage_dir, vq_model_instance):
    """
    为当前对话行初始化数据库。
    注意：在此脚本中，特定目录下的文件应该已经被 copy_existing_databases 准备好了。
    """
    meta_path = os.path.join(specific_storage_dir, "metadata.db")
    vec_path = os.path.join(specific_storage_dir, "vector_stores")

    if has_non_ascii_path(os.path.abspath(vec_path)):
        raise ValueError(
            f"向量目录包含非 ASCII 路径，FAISS 在 Windows 下可能无法读取: {os.path.abspath(vec_path)}"
        )

    os.makedirs(vec_path, exist_ok=True)

    # 初始化 DB Handler (会自动加载已存在的DB)
    meta_db = MetadataDBHandler(db_path=meta_path)
    vec_db = VectorDBHandler(storage_path=vec_path)

    # 加载索引 (确保读取刚才复制过来的 index)
    vec_db.create_or_load_index("atomic_memory", Model_Dimension)
    loaded_index = vec_db.indexes.get("atomic_memory", {}).get("index")
    loaded_map = vec_db.indexes.get("atomic_memory", {}).get("id_map", {})
    if loaded_index is not None:
        print(f"  [Vector Check] atomic_memory.ntotal = {loaded_index.ntotal} | id_map = {len(loaded_map)}")

    # 初始化 Pipeline
    pipeline = AsyncProcessingPipeline(splitter, embedder, llm_interface, meta_db, vec_db, vq_model_instance)
    query_engine = QueryEngine(embedder, meta_db, vec_db, llm_interface, vq_model_instance)

    return pipeline, query_engine, meta_db, vec_db


def save_qa_trace_log(trace_data: Dict[str, Any], log_dir: str, q_id: str):
    """保存单条 QA 的完整追踪日志"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_path = os.path.join(log_dir, f"qa_trace_{q_id}.json")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save trace log for {q_id}: {e}")

async def run_qa_evaluation(query_engine, judge, dataset_type, question, correct_answer, question_id, fact_num,
                            question_type="standard", log_dir=None, query_date=None):
    """QA 执行逻辑 (保持不变)"""
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

        # Query
        query_result = query_engine.query(
            query_text,
            total_atomic_count=fact_num,
            k=20,
            fusion_strategy='llm',
            enable_hyde=False,
            query_date=query_date
        )

        final_prompt = query_result["final_prompt"]
        trace_info = query_result["trace_info"]

        full_trace_record["retrieval_trace"] = trace_info

        # Generate
        llm_qa = OpenAIInterface(api_key=OPENAI_API_KEY, model=QA_LLM_MODEL, base_url=BaseUrl)

        generated_answer = llm_qa.get_completion(final_prompt)
        qa_end_time = time.time()
        perf_stats["latency_seconds"] = qa_end_time - qa_start_time
        usage_qa = llm_qa.get_total_usage() or {}
        perf_stats["generation_tokens"] = usage_qa.get("total_tokens", 0)
        perf_stats["generation_prompt_tokens"] = usage_qa.get("total_prompt_tokens", 0)
        perf_stats["generation_completion_tokens"] = usage_qa.get("total_completion_tokens", 0)


        result_entry["generated_answer"] = generated_answer
        print(f"    生成的答案: {generated_answer}")

        full_trace_record["generation"] = {
            "final_prompt_used": final_prompt,
            "model_answer": generated_answer,
            "ground_truth": correct_answer
        }

        # Judge
        if dataset_type == "locomo":
            judge_category = map_locomo_type_to_judge_category(question_type)
        else:
            judge_category = map_question_type_to_judge_category(question_type)

        result_entry["judge_category"] = judge_category

        if judge_category == LLMJudge.ABSTENTION_TASK or 'abs' in str(question_id):
            is_correct = judge.evaluate(judge_category, question, generated_answer, explanation=correct_answer, query_date=query_date)
        elif judge_category == LLMJudge.PREFERENCE_TASK:
            is_correct = judge.evaluate(judge_category, question, generated_answer, rubric=correct_answer, query_date=query_date)
        else:
            is_correct = judge.evaluate(judge_category, question, generated_answer, correct_answer=correct_answer, query_date=query_date)

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
        print(f"QA Error: {e}")
        import traceback
        traceback.print_exc()

    result_entry["perf_stats"] = perf_stats
    return result_entry


async def main_async():
    # 1. 加载数据
    if DATASET_TYPE == "locomo":
        test_data = load_locomo_dataset(DATA_PATH)
    else:
        test_data = load_longmemeval_dataset(DATA_PATH)

    if not test_data:
        print("未加载到数据，退出。")
        return

    from release_runtime import validate_source_memory
    validate_source_memory(CONFIG, len(test_data))

    # 自动续跑: 找到第一个未完成 row，并从它的前一条开始重跑
    if has_non_ascii_path(TEST_ROOT_DIR):
        raise ValueError(f"TEST_ROOT_DIR 不能包含中文或其他非 ASCII 字符: {TEST_ROOT_DIR}")

    os.makedirs(TEST_ROOT_DIR, exist_ok=True)

    from release_runtime import write_run_manifest
    write_run_manifest(CONFIG)

    resume_start_idx = find_resume_start_index(test_data)
    if resume_start_idx >= len(test_data):
        print(f"检测到 {TEST_ROOT_DIR} 中所有 row 都已有 qa_trace，默认从最后一条开始不再重跑。")
        resume_start_idx = len(test_data)
    elif os.listdir(TEST_ROOT_DIR):
        print(f"检测到已有中间结果，将从 row_{resume_start_idx}_data 开始重新运行。")
        cleanup_rows_from_index(resume_start_idx)
    else:
        print("未检测到可续跑结果，将从 row_0_data 全量开始运行。")

    print(f"源数据库路径 (读取): {SOURCE_ROOT_DIR}")
    print(f"新测试输出路径 (写入): {TEST_ROOT_DIR}")

    # 2. 初始化静态组件
    splitter, embedder, llm_interface, judge_llm_interface, judge, vq_model_static = init_components_once()

    results = load_existing_results(test_data, resume_start_idx, DATASET_TYPE)
    if results:
        print(f"  > [Resume] 已从现有 qa_trace 载入 {len(results)} 条已完成结果。")
    # 记录每个对话的演化耗时（秒）
    evolution_times = []
    accum_memory_prompt_tokens = 0
    accum_memory_completion_tokens = 0
    accum_memory_tokens = 0
    accum_memory_calls = 0

    accum_evolution_prompt_tokens = 0
    accum_evolution_completion_tokens = 0
    accum_evolution_tokens = 0
    accum_evolution_calls = 0

    accum_qa_prompt_tokens = 0
    accum_qa_completion_tokens = 0
    accum_qa_tokens = 0

    for existing_result in results:
        stats = existing_result.get("perf_stats", {})
        accum_qa_tokens += stats.get("generation_tokens", 0)
        accum_qa_prompt_tokens += stats.get("generation_prompt_tokens", 0)
        accum_qa_completion_tokens += stats.get("generation_completion_tokens", 0)

    print(f"\n=== 开始测试 (Skip Ingestion Mode) ===")

    if resume_start_idx >= len(test_data):
        print("没有待处理的新 row，将直接汇总当前已有结果。")

    for idx, item in enumerate(test_data[resume_start_idx:], start=resume_start_idx):
        print(f"\nProcessing Row {idx + 1}/{len(test_data)} ...")

        unique_folder_name = f"row_{idx}_data"

        # 源路径 (用于复制)
        source_row_storage_path = os.path.join(SOURCE_ROOT_DIR, unique_folder_name)
        # 目标路径 (用于使用和日志)
        current_row_storage_path = os.path.join(TEST_ROOT_DIR, unique_folder_name)

        os.makedirs(current_row_storage_path, exist_ok=True)

        # [关键步骤] 复制现有数据库
        copy_existing_databases(source_row_storage_path, current_row_storage_path)

        # 初始化 Pipeline (基于复制过来的DB)
        pipeline, query_engine, meta_db, vec_db = init_databases_and_pipeline(
            splitter, embedder, llm_interface,
            specific_storage_dir=current_row_storage_path,
            vq_model_instance=vq_model_static
        )

        try:
            # 数据处理部分 (已全部跳过)
            print("  > [Info] Skipping memory ingestion phase (using existing databases).")

            # 仅统计当前数据库中的事实数量
            count_fact = len(meta_db.get_all_active_atomics())
            print(f"  > Loaded Active Fact Count from DB: {count_fact}")

            # QA 评估部分
            if item["dataset_type"] == "locomo":
                source_id = item["row_id"]

                for q_idx, qa_pair in enumerate(item["qa_list"]):
                    q_id = f"{source_id}_q{q_idx}"
                    q_category = qa_pair.get('category', 1)

                    if SKIP_LOCOMO_ADVERSARIAL and q_category == 5:
                        print(f"    [Config] Skipping Adversarial Question (ID: {q_id}, Category 5)...")
                        continue

                    ground_truth = qa_pair.get('answer')
                    if q_category == 5:
                        distractor = qa_pair.get('adversarial_answer', '')
                        ground_truth = f"Not Answerable. (Note: The dataset provides a distractor/adversarial answer: '{distractor}')"
                    if ground_truth is None:
                        ground_truth = str(qa_pair.get('adversarial_answer', 'Unknown'))
                    else:
                        ground_truth = str(ground_truth)

                    print("=" * 60 + '\n')
                    print(f"    Q Type: {q_category} | GroundTruth: {ground_truth}")

                    res = await run_qa_evaluation(
                        query_engine, judge, "locomo",
                        qa_pair['question'],
                        ground_truth,
                        q_id,
                        count_fact,
                        question_type=q_category,
                        log_dir=current_row_storage_path,
                        query_date=None # Locomo 无时间戳
                    )
                    results.append(res)
                    stats = res.get("perf_stats", {})
                    accum_qa_tokens += stats.get("generation_tokens", 0)
                    accum_qa_prompt_tokens += stats.get("generation_prompt_tokens", 0)
                    accum_qa_completion_tokens += stats.get("generation_completion_tokens", 0)

            else:
                # LongMemEval 逻辑
                timestamps = item["haystack_dates"]
                # Log the question reference timestamp.
                if timestamps:
                     print(f"  > [Time Check] First timestamp in source data: {timestamps[0]}")

                q_id = item["question_id"]
                question_reference_time = item.get("question_date")
                print(f"  > Querying: {item['question']} [RefTime: {question_reference_time}]")


                # Optional asynchronous consolidation.
                print(f"\n  >>> [Optimization] 启动异步记忆演化 (Memory Evolution)...")

                usage_before_evo = llm_interface.get_total_usage() or {}
                tokens_before_evo = usage_before_evo.get('total_tokens', 0)
                prompts_before_evo = usage_before_evo.get('total_prompt_tokens', 0)
                comps_before_evo = usage_before_evo.get('total_completion_tokens', 0)
                calls_before_evo = getattr(llm_interface, 'call_count', 0)

                # 执行演化
                start_evo = time.time()

                # 初始化 Consolidator
                consolidator = MemoryConsolidator(
                    meta_db=meta_db,
                    vec_db=vec_db,
                    fact_num=count_fact,
                    embedder=embedder,
                    llm=llm_interface,
                    vq_model=vq_model_static,
                    max_concurrency=10  # 建议根据 API 限流情况调整，8-16 之间
                )


                await consolidator.execute_evolution_async()
                evo_elapsed = time.time() - start_evo
                evolution_times.append(evo_elapsed)
                usage_after_evo = llm_interface.get_total_usage() or {}
                tokens_after_evo = usage_after_evo.get('total_tokens', 0)
                prompts_after_evo = usage_after_evo.get('total_prompt_tokens', 0)
                comps_after_evo = usage_after_evo.get('total_completion_tokens', 0)
                calls_after_evo = getattr(llm_interface, 'call_count', 0)

                evo_tokens_delta = tokens_after_evo - tokens_before_evo
                evo_prompt_delta = prompts_after_evo - prompts_before_evo
                evo_completion_delta = comps_after_evo - comps_before_evo
                evo_calls_delta = calls_after_evo - calls_before_evo

                accum_evolution_tokens += evo_tokens_delta
                accum_evolution_prompt_tokens += evo_prompt_delta
                accum_evolution_completion_tokens += evo_completion_delta
                accum_evolution_calls += evo_calls_delta
                print(f"  >>> [Optimization] 演化耗时: {evo_elapsed:.2f} 秒")
                print(
                    f"  >>> [Optimization] 演化 Token: Prompt {evo_prompt_delta} | "
                    f"Completion {evo_completion_delta} | Total {evo_tokens_delta} | Calls {evo_calls_delta}"
                )

                print(f"  >>> [Optimization] 演化流程完成。")

                # 再次统计数量 (查看演化后的变化)
                count_fact = len(meta_db.get_all_active_atomics())
                print(f"  > Active Fact Count (Post-Evolution): {count_fact}")


                res = await run_qa_evaluation(
                    query_engine, judge, "longmemeval",
                    item['question'], item['answer'], q_id, count_fact,
                    item.get('question_type', 'standard'),
                    log_dir=current_row_storage_path,
                    query_date=question_reference_time # <--- 传入时间戳
                )
                results.append(res)
                stats = res.get("perf_stats", {})
                accum_qa_tokens += stats.get("generation_tokens", 0)
                accum_qa_prompt_tokens += stats.get("generation_prompt_tokens", 0)
                accum_qa_completion_tokens += stats.get("generation_completion_tokens", 0)

        finally:
            pipeline.shutdown_databases()

    # 统计报告
    print("\n" + "=" * 60)
    print("详细统计报告 (Detailed Statistics) - Skip Ingestion Mode")
    print("=" * 60)

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
        if raw_cat not in stats_by_category:
            stats_by_category[raw_cat] = {"total": 0, "correct": 0}
        stats_by_category[raw_cat]["total"] += 1
        if r["is_correct"]:
            stats_by_category[raw_cat]["correct"] += 1

    sorted_cats = sorted(stats_by_category.keys(), key=lambda x: str(x))
    for cat in sorted_cats:
        s = stats_by_category[cat]
        acc = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0.0
        display_name = str(cat)
        if DATASET_TYPE == "locomo" and isinstance(cat, int) and cat in locomo_cat_names:
            display_name = f"Category {cat} [{locomo_cat_names[cat]}]"
        elif DATASET_TYPE == "locomo" and str(cat).isdigit():
            c_int = int(cat)
            if c_int in locomo_cat_names:
                display_name = f"Category {c_int} [{locomo_cat_names[c_int]}]"

        print(f"类别: {display_name:<35} | 总数: {s['total']:<4} | 正确: {s['correct']:<4} | 准确率: {acc:.2f}%")

    print("-" * 60)
    judge_usage = judge_llm_interface.get_total_usage() or {}
    accum_judge_prompt_tokens = judge_usage.get('total_prompt_tokens', 0)
    accum_judge_completion_tokens = judge_usage.get('total_completion_tokens', 0)
    accum_judge_tokens = judge_usage.get('total_tokens', 0)
    grand_total_tokens = accum_memory_tokens + accum_evolution_tokens + accum_qa_tokens + accum_judge_tokens
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
    print("=" * 60)

    # 打印演化平均耗时汇总
    if evolution_times:
        avg_evo = sum(evolution_times) / len(evolution_times)
        print(f"平均演化耗时 (每对话): {avg_evo:.2f} 秒，样本数: {len(evolution_times)}")
    else:
        print("平均演化耗时: N/A（没有执行任何演化）")

    output_file = os.path.join(TEST_ROOT_DIR, f"eval_results_{DATASET_TYPE}_skipped.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"详细结果Json已保存到: {output_file}")

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
        "completion_tokens_total": grand_total_completion_tokens
    }
    with open(os.path.join(TEST_ROOT_DIR, f"eval_performance_{DATASET_TYPE}_skipped.json"), 'w', encoding='utf-8') as f:
        json.dump(perf_log, f, indent=2)

    print(f"新的完整日志目录: {TEST_ROOT_DIR}")


if __name__ == '__main__':
    asyncio.run(main_async())
