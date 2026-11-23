

import os
import json
import shutil
import asyncio
import time  # For timing
from typing import List, Dict, Any, Union
from datetime import datetime
import nltk

# 导入项目模块
from src.core.metadata_db_handler import MetadataDBHandler
from src.core.vector_db_handler import VectorDBHandler
from src.processing.text_splitter import ConversationSplitter

from src.utils.embedding_bge import EmbeddingModel
from src.utils.vqvae_inference_v10 import ModelInference  # 确保引用正确

from src.processing.longmemeval_memory_pipeline import AsyncProcessingPipeline
from release_runtime import selected_query_engine
QueryEngine = selected_query_engine()
from src.utils.llm_interface import OpenAIInterface
from src.utils.llm_judge import LLMJudge

# 引入演化模块
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
    """Map LoCoMo categories to evaluator tasks.

    Category 2 uses temporal evaluation; category 5 uses abstention.
    Categories 1, 3 and 4 use the standard answer-matching prompt."""
    try:
        cat_id = int(question_type)
    except (ValueError, TypeError):
        # 如果传入的是字符串描述而非数字，尝试关键词匹配作为 fallback
        q_type_str = str(question_type).lower().strip()
        if "temporal" in q_type_str: return LLMJudge.TEMPORAL_REASONING
        if "adversarial" in q_type_str: return LLMJudge.ABSTENTION_TASK
        return LLMJudge.STANDARD_TASK

    if cat_id == 2:
        return LLMJudge.TEMPORAL_REASONING
    elif cat_id == 5:
        return LLMJudge.ABSTENTION_TASK
    else:
        # 1, 3, 4 均归类为 Standard 任务
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
            "row_id": f"locomo_row_{row_idx}",  # 用作文件夹名称的一部分
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

    # Monkey Patch: 为 llm_interface 添加调用计数功能
    # 这允许我们在不修改 OpenAIInterface 源码的情况下统计调用次数
    if not hasattr(llm_interface, 'call_count'):
        llm_interface.call_count = 0
        original_completion = llm_interface.get_completion

        def wrapped_completion(*args, **kwargs):
            llm_interface.call_count += 1
            return original_completion(*args, **kwargs)

        llm_interface.get_completion = wrapped_completion
        print(">> [System] 已启用 LLM 调用计数器 (Monkey Patch)")

    judge_llm_interface = OpenAIInterface(api_key=OPENAI_API_KEY, model=JUDGE_LLM_MODEL, base_url=BaseUrl)
    judge = LLMJudge(llm_interface=judge_llm_interface)

    print("正在初始化 VQ-VAE (Static)...")
    vq_model_static = ModelInference(device_str=CONFIG.device)
    return splitter, embedder, llm_interface, judge_llm_interface, judge, vq_model_static


def extract_row_index(folder_name: str) -> int:
    try:
        parts = folder_name.split('_')
        if len(parts) >= 3 and parts[0] == 'row' and parts[2] == 'data':
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
            return idx
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


def _build_result_entry_from_trace(trace_obj: Dict[str, Any]) -> Dict[str, Any]:
    generation = trace_obj.get("generation", {})
    evaluation = trace_obj.get("evaluation", {})
    return {
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


def load_existing_results(stop_before_idx: int) -> List[Dict[str, Any]]:
    loaded_results = []

    for idx in range(stop_before_idx):
        row_dir = os.path.join(TEST_ROOT_DIR, f"row_{idx}_data")
        if not os.path.isdir(row_dir):
            continue

        row_metrics = {}
        metrics_path = os.path.join(row_dir, "row_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    row_metrics = json.load(f)
            except Exception as e:
                print(f"  [Resume] 读取 row_metrics 失败 {metrics_path}: {e}")
                row_metrics = {}

        trace_paths = sorted(
            os.path.join(row_dir, name)
            for name in os.listdir(row_dir)
            if name.startswith("qa_trace_") and name.endswith(".json")
        )

        for trace_path in trace_paths:
            try:
                with open(trace_path, 'r', encoding='utf-8') as f:
                    trace_obj = json.load(f)
                result_entry = _build_result_entry_from_trace(trace_obj)

                qa_metrics = row_metrics.get("qa", {})
                qa_times = qa_metrics.get("latency_seconds_list", [])
                if len(trace_paths) == 1:
                    result_entry["perf_stats"] = {
                        "latency_seconds": qa_times[0] if qa_times else 0.0,
                        "generation_tokens": qa_metrics.get("total", 0),
                        "generation_prompt_tokens": qa_metrics.get("prompt", 0),
                        "generation_completion_tokens": qa_metrics.get("completion", 0)
                    }

                loaded_results.append(result_entry)
            except Exception as e:
                print(f"  [Resume] 跳过损坏的 trace 文件 {trace_path}: {e}")

    return loaded_results


def save_row_metrics_log(row_metrics: Dict[str, Any], row_dir: str):
    file_path = os.path.join(row_dir, "row_metrics.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(row_metrics, f, ensure_ascii=False, indent=2)


def load_existing_row_metrics(stop_before_idx: int) -> Dict[str, Any]:
    restored = {
        "memory_prompt_tokens": 0,
        "memory_completion_tokens": 0,
        "memory_tokens": 0,
        "memory_calls": 0,
        "evolution_prompt_tokens": 0,
        "evolution_completion_tokens": 0,
        "evolution_tokens": 0,
        "evolution_calls": 0,
        "qa_prompt_tokens": 0,
        "qa_completion_tokens": 0,
        "qa_tokens": 0,
        "memory_times": [],
        "evolution_times": [],
        "qa_times": [],
        "missing_rows": []
    }

    for idx in range(stop_before_idx):
        row_dir = os.path.join(TEST_ROOT_DIR, f"row_{idx}_data")
        metrics_path = os.path.join(row_dir, "row_metrics.json")
        if not os.path.exists(metrics_path):
            restored["missing_rows"].append(idx)
            continue

        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                row_metrics = json.load(f)
        except Exception as e:
            print(f"  [Resume] 读取 row_metrics 失败 {metrics_path}: {e}")
            restored["missing_rows"].append(idx)
            continue

        memory = row_metrics.get("memory", {})
        evolution = row_metrics.get("evolution", {})
        qa = row_metrics.get("qa", {})

        restored["memory_prompt_tokens"] += memory.get("prompt", 0)
        restored["memory_completion_tokens"] += memory.get("completion", 0)
        restored["memory_tokens"] += memory.get("total", 0)
        restored["memory_calls"] += memory.get("calls", 0)

        restored["evolution_prompt_tokens"] += evolution.get("prompt", 0)
        restored["evolution_completion_tokens"] += evolution.get("completion", 0)
        restored["evolution_tokens"] += evolution.get("total", 0)
        restored["evolution_calls"] += evolution.get("calls", 0)

        restored["qa_prompt_tokens"] += qa.get("prompt", 0)
        restored["qa_completion_tokens"] += qa.get("completion", 0)
        restored["qa_tokens"] += qa.get("total", 0)

        mem_time = memory.get("duration_seconds")
        evo_time = evolution.get("duration_seconds")
        qa_times = qa.get("latency_seconds_list", [])

        if isinstance(mem_time, (int, float)):
            restored["memory_times"].append(mem_time)
        if isinstance(evo_time, (int, float)):
            restored["evolution_times"].append(evo_time)
        if isinstance(qa_times, list):
            restored["qa_times"].extend(
                t for t in qa_times
                if isinstance(t, (int, float))
            )

    return restored


def init_databases_and_pipeline(splitter, embedder, llm_interface, specific_storage_dir, vq_model_instance):
    """
    为当前对话行初始化数据库。
    数据存储在 specific_storage_dir 中，互不干扰。
    """
    # 1. 定义该行独有的数据库路径
    meta_path = os.path.join(specific_storage_dir, "metadata.db")
    vec_path = os.path.join(specific_storage_dir, "vector_stores")

    # 确保目录存在
    os.makedirs(vec_path, exist_ok=True)

    # 2. 初始化 DB Handler
    meta_db = MetadataDBHandler(db_path=meta_path)
    vec_db = VectorDBHandler(storage_path=vec_path)

    # 3. 创建索引
    vec_db.create_or_load_index("atomic_memory", Model_Dimension)
    # vec_db.create_or_load_index("summary_memory", 384) # 摘要层已移除

    # 4. 初始化 Pipeline
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
    """QA 执行逻辑 (增强版：包含完整 Trace 记录 + 时间感知)"""

    # Start timing for QA latency
    qa_start_time = time.time()

    result_entry = {
        "question_id": question_id,
        "question": question,
        "ground_truth": correct_answer,
        "generated_answer": "",
        "is_correct": False,
        "judge_category": "",
        "raw_category": question_type,  # 记录原始分类以便后续统计
        "error": None
    }

    # Performance stats container
    perf_stats = {
        "latency_seconds": 0.0,
        "generation_tokens": 0,
        "generation_prompt_tokens": 0,
        "generation_completion_tokens": 0
    }

    # 初始化 Trace 记录结构
    full_trace_record = {
        "question_id": question_id,
        "timestamp": datetime.now().isoformat(),
        "dataset_type": dataset_type,
        "question": question,
        "question_type_raw": question_type,
        "query_reference_time": query_date,  # 记录参考时间
        "retrieval_trace": {},
        "generation": {},
        "evaluation": {}
    }

    try:
        query_text = question
        if dataset_type != "locomo":
            # 将LongMemEval中的I和my替换为the user和the user's
            query_text = query_text.replace(" I ", " the user ").replace(" my ", " the user's ")

        # [Step 1] 执行查询 (Query) - 传入时间感知参数
        query_result = query_engine.query(
            query_text,
            total_atomic_count=fact_num,
            k=10,
            fusion_strategy='llm',  # 可选 'rrf' 或 'llm'
            enable_hyde=False,  # 可选 True 开启假设答案增强
            query_date=query_date  # <---  传入参考时间，用于生成 Prompt
        )

        final_prompt = query_result["final_prompt"]
        trace_info = query_result["trace_info"]

        # 记录检索过程
        full_trace_record["retrieval_trace"] = trace_info

        # [Step 2] 生成答案 (Generate)
        llm_qa = OpenAIInterface(api_key=OPENAI_API_KEY, model=QA_LLM_MODEL, base_url=BaseUrl)

        # Use asyncio.to_thread for IO-bound LLM call
        generated_answer = await asyncio.to_thread(llm_qa.get_completion, final_prompt)

        # Capture tokens and timing right after generation
        qa_end_time = time.time()
        perf_stats["latency_seconds"] = qa_end_time - qa_start_time
        usage_qa = llm_qa.get_total_usage() or {}
        perf_stats["generation_tokens"] = usage_qa.get("total_tokens", 0)
        perf_stats["generation_prompt_tokens"] = usage_qa.get("total_prompt_tokens", 0)
        perf_stats["generation_completion_tokens"] = usage_qa.get("total_completion_tokens", 0)

        result_entry["generated_answer"] = generated_answer
        print(f"    生成的答案: {generated_answer}")

        # 记录生成信息
        full_trace_record["generation"] = {
            "final_prompt_used": final_prompt,
            "model_answer": generated_answer,
            "ground_truth": correct_answer
        }

        # [Step 3] 评判 (Judge) - 动态映射逻辑
        if dataset_type == "locomo":
            judge_category = map_locomo_type_to_judge_category(question_type)
        else:
            judge_category = map_question_type_to_judge_category(question_type)

        result_entry["judge_category"] = judge_category

        def run_judge_logic():
            if judge_category == LLMJudge.ABSTENTION_TASK or 'abs' in str(question_id):
                return judge.evaluate(
                    judge_category,
                    question,
                    generated_answer,
                    explanation=correct_answer,
                    query_date=query_date
                )
            elif judge_category == LLMJudge.PREFERENCE_TASK:
                return judge.evaluate(
                    judge_category,
                    question,
                    generated_answer,
                    rubric=correct_answer,
                    query_date=query_date
                )
            else:
                return judge.evaluate(
                    judge_category,
                    question,
                    generated_answer,
                    correct_answer=correct_answer,
                    query_date=query_date
                )

        # [Async] Run judge in thread
        is_correct = await asyncio.to_thread(run_judge_logic)

        result_entry["is_correct"] = is_correct

        # 记录评判结果
        full_trace_record["evaluation"] = {
            "is_correct": is_correct,
            "category": judge_category,
            "judge_explanation_used": correct_answer,
            "judge_reference_time_used": query_date
        }

        # [Step 4] 保存 Trace 日志
        if log_dir:
            save_qa_trace_log(full_trace_record, log_dir, question_id)

    except Exception as e:
        result_entry["error"] = str(e)
        print(f"QA Error: {e}")
        import traceback
        traceback.print_exc()

    # Attach performance stats
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

    os.makedirs(TEST_ROOT_DIR, exist_ok=True)

    from release_runtime import write_run_manifest
    write_run_manifest(CONFIG)

    resume_start_idx = find_resume_start_index(test_data)
    has_existing_rows = any(
        extract_row_index(entry) >= 0
        for entry in os.listdir(TEST_ROOT_DIR)
    )

    if resume_start_idx >= len(test_data):
        print(f"检测到 {TEST_ROOT_DIR} 中所有 row 都已有 qa_trace，默认不重跑。")
    elif has_existing_rows:
        print(f"检测到已有中间结果，将从 row_{resume_start_idx}_data 开始重新运行。")
        cleanup_rows_from_index(resume_start_idx)
    else:
        print("未检测到可续跑结果，将从 row_0_data 全量开始运行。")

    # Global performance accumulators (分离统计)
    accum_memory_prompt_tokens = 0
    accum_memory_completion_tokens = 0
    accum_memory_tokens = 0
    accum_memory_calls = 0

    accum_evolution_prompt_tokens = 0
    accum_evolution_completion_tokens = 0
    accum_evolution_tokens = 0  # Evolution stats
    accum_evolution_calls = 0

    accum_qa_prompt_tokens = 0
    accum_qa_completion_tokens = 0
    accum_qa_tokens = 0

    all_memory_times = []
    all_evolution_times = []
    all_qa_times = []

    # 2. 初始化静态组件
    splitter, embedder, llm_interface, judge_llm_interface, judge, vq_model_static = init_components_once()

    restored_metrics = load_existing_row_metrics(resume_start_idx)
    missing_metric_rows = restored_metrics.pop("missing_rows", [])

    accum_memory_prompt_tokens += restored_metrics["memory_prompt_tokens"]
    accum_memory_completion_tokens += restored_metrics["memory_completion_tokens"]
    accum_memory_tokens += restored_metrics["memory_tokens"]
    accum_memory_calls += restored_metrics["memory_calls"]

    accum_evolution_prompt_tokens += restored_metrics["evolution_prompt_tokens"]
    accum_evolution_completion_tokens += restored_metrics["evolution_completion_tokens"]
    accum_evolution_tokens += restored_metrics["evolution_tokens"]
    accum_evolution_calls += restored_metrics["evolution_calls"]

    accum_qa_prompt_tokens += restored_metrics["qa_prompt_tokens"]
    accum_qa_completion_tokens += restored_metrics["qa_completion_tokens"]
    accum_qa_tokens += restored_metrics["qa_tokens"]

    all_memory_times.extend(restored_metrics["memory_times"])
    all_evolution_times.extend(restored_metrics["evolution_times"])
    all_qa_times.extend(restored_metrics["qa_times"])

    results = load_existing_results(resume_start_idx)
    if results:
        print(f"  > [Resume] 已从现有 qa_trace 载入 {len(results)} 条已完成结果。")
    if missing_metric_rows:
        preview = ", ".join(f"row_{idx}" for idx in missing_metric_rows[:10])
        more_suffix = " ..." if len(missing_metric_rows) > 10 else ""
        print(
            f"  > [Resume] 以下已完成 row 缺少 row_metrics，续跑后的总耗时/Token 汇总不会包含它们: "
            f"{preview}{more_suffix}"
        )

    print(f"\n=== 开始测试 (数据结果保留在 {TEST_ROOT_DIR} 下的各个子目录中) ===")

    if resume_start_idx >= len(test_data):
        print("没有待处理的新 row，将直接汇总当前已有结果。")

    for idx, item in enumerate(test_data):
        if idx < resume_start_idx:
            continue

        print(f"\nProcessing Row {idx + 1}/{len(test_data)} ...")

        # [核心] 为当前行生成唯一的子目录
        unique_folder_name = f"row_{idx}_data"
        current_row_storage_path = os.path.join(TEST_ROOT_DIR, unique_folder_name)

        os.makedirs(current_row_storage_path, exist_ok=True)
        print(f"  > 数据库存储路径: {current_row_storage_path}")

        # 初始化 Pipeline
        pipeline, query_engine, meta_db, vec_db = init_databases_and_pipeline(
            splitter, embedder, llm_interface,
            specific_storage_dir=current_row_storage_path,
            vq_model_instance=vq_model_static  # <--- 传进去
        )

        try:
            row_qa_prompt_tokens = 0
            row_qa_completion_tokens = 0
            row_qa_tokens = 0
            row_qa_times = []

            # ==========================================
            # 阶段 1: 记忆构建 (Memory Construction)
            # ==========================================
            t_mem_start = time.time()
            # 获取构建前的状态快照
            usage_before_mem = llm_interface.get_total_usage() or {}
            tokens_before_mem = usage_before_mem.get('total_tokens', 0)
            prompts_before_mem = usage_before_mem.get('total_prompt_tokens', 0)
            comps_before_mem = usage_before_mem.get('total_completion_tokens', 0)
            calls_before_mem = getattr(llm_interface, 'call_count', 0)

            # 数据处理
            if item["dataset_type"] == "locomo":
                source_id = item["row_id"]
                # 传入 log_dir 以生成提取过程的 trace (Process Trace)
                await pipeline.process_locomo_data_async(
                    item["conversation"],
                    source_id,
                    log_dir=current_row_storage_path
                )
            else:
                # LongMemEval 逻辑：移除外部循环，使用全量处理方法
                sessions = item["haystack_sessions"]
                print(f"  > 历史会话数量为: {len(sessions)}")
                timestamps = item["haystack_dates"]
                q_id = item["question_id"]

                await pipeline.process_longmemeval_data_async(
                    sessions, timestamps, q_id, log_dir=current_row_storage_path
                )

            # Memory Construction End & Stats Calculation
            t_mem_end = time.time()
            usage_after_mem = llm_interface.get_total_usage() or {}
            tokens_after_mem = usage_after_mem.get('total_tokens', 0)
            prompts_after_mem = usage_after_mem.get('total_prompt_tokens', 0)
            comps_after_mem = usage_after_mem.get('total_completion_tokens', 0)
            calls_after_mem = getattr(llm_interface, 'call_count', 0)

            mem_duration = t_mem_end - t_mem_start
            mem_tokens_delta = tokens_after_mem - tokens_before_mem
            mem_prompt_delta = prompts_after_mem - prompts_before_mem
            mem_completion_delta = comps_after_mem - comps_before_mem
            mem_calls_delta = calls_after_mem - calls_before_mem

            # 累加
            accum_memory_tokens += mem_tokens_delta
            accum_memory_prompt_tokens += mem_prompt_delta
            accum_memory_completion_tokens += mem_completion_delta
            accum_memory_calls += mem_calls_delta
            all_memory_times.append(mem_duration)

            count_fact = meta_db.get_total_atomic_count()
            print(f"  > [Build] Facts: {count_fact} | Time: {mem_duration:.2f}s | Prompt: {mem_prompt_delta} | Completion: {mem_completion_delta} | Total: {mem_tokens_delta}")

            # ==========================================
            # Stage 2: optional consolidation controlled by the run profile.
            # ==========================================
            # [计时与统计 - 演化]
            t_evo_start = time.time()
            usage_before_evo = llm_interface.get_total_usage() or {}
            tokens_before_evo = usage_before_evo.get('total_tokens', 0)
            prompts_before_evo = usage_before_evo.get('total_prompt_tokens', 0)
            comps_before_evo = usage_before_evo.get('total_completion_tokens', 0)
            calls_before_evo = getattr(llm_interface, 'call_count', 0)


            if ENABLE_CONSOLIDATION:
                print("\n  >>> [Optimization] Starting async memory evolution...")
                consolidator = MemoryConsolidator(
                    meta_db=meta_db,
                    vec_db=vec_db,
                    fact_num=count_fact,
                    embedder=embedder,
                    llm=llm_interface,
                    vq_model=vq_model_static,
                    max_concurrency=10,
                )
                await consolidator.execute_evolution_async()
                print("  >>> [Optimization] Memory evolution completed.")
                count_fact_final = meta_db.get_total_atomic_count()
                count_fact = count_fact_final
                print(f"  > [Evo] Facts Post-Evolution: {count_fact_final}")

            # [统计] 演化结束 (即使注释掉也要计算，结果为 0)
            t_evo_end = time.time()
            usage_after_evo = llm_interface.get_total_usage() or {}
            tokens_after_evo = usage_after_evo.get('total_tokens', 0)
            prompts_after_evo = usage_after_evo.get('total_prompt_tokens', 0)
            comps_after_evo = usage_after_evo.get('total_completion_tokens', 0)
            calls_after_evo = getattr(llm_interface, 'call_count', 0)

            evo_duration = t_evo_end - t_evo_start
            evo_tokens_delta = tokens_after_evo - tokens_before_evo
            evo_prompt_delta = prompts_after_evo - prompts_before_evo
            evo_completion_delta = comps_after_evo - comps_before_evo
            evo_calls_delta = calls_after_evo - calls_before_evo

            accum_evolution_tokens += evo_tokens_delta
            accum_evolution_prompt_tokens += evo_prompt_delta
            accum_evolution_completion_tokens += evo_completion_delta
            accum_evolution_calls += evo_calls_delta
            all_evolution_times.append(evo_duration)

            # ==========================================
            # 阶段 3: 问答 (QA)
            # ==========================================
            if item["dataset_type"] == "locomo":
                # LoCoMo QA Logic
                print(f"\n  >>> 启动并发 QA ...")

                for q_idx, qa_pair in enumerate(item["qa_list"]):

                    q_id = f"{source_id}_q{q_idx}"
                    q_category = qa_pair.get('category', 1)

                    if SKIP_LOCOMO_ADVERSARIAL and q_category == 5: continue

                    ground_truth = str(qa_pair.get('answer'))
                    if q_category == 5 and not ground_truth:
                        ground_truth = str(qa_pair.get('adversarial_answer', 'Unknown'))

                    res = await run_qa_evaluation(
                        query_engine, judge, "locomo",
                        qa_pair['question'], ground_truth, q_id, count_fact,
                        question_type=q_category, log_dir=current_row_storage_path
                    )
                    results.append(res)

                    # Capture QA stats
                    stats = res.get("perf_stats", {})
                    all_qa_times.append(stats.get("latency_seconds", 0))
                    accum_qa_tokens += stats.get("generation_tokens", 0)
                    accum_qa_prompt_tokens += stats.get("generation_prompt_tokens", 0)
                    accum_qa_completion_tokens += stats.get("generation_completion_tokens", 0)
                    row_qa_times.append(stats.get("latency_seconds", 0))
                    row_qa_tokens += stats.get("generation_tokens", 0)
                    row_qa_prompt_tokens += stats.get("generation_prompt_tokens", 0)
                    row_qa_completion_tokens += stats.get("generation_completion_tokens", 0)

            else:
                # LongMemEval QA Logic
                print(f"  > Querying: {item['question']}")

                # 提取问题时间
                question_reference_time = item.get("question_date")

                res = await run_qa_evaluation(
                    query_engine, judge, "longmemeval",
                    item['question'], item['answer'], item["question_id"], count_fact,
                    item.get('question_type', 'standard'),
                    log_dir=current_row_storage_path,
                    query_date=question_reference_time
                )
                results.append(res)

                # Capture QA stats
                stats = res.get("perf_stats", {})
                all_qa_times.append(stats.get("latency_seconds", 0))
                accum_qa_tokens += stats.get("generation_tokens", 0)
                accum_qa_prompt_tokens += stats.get("generation_prompt_tokens", 0)
                accum_qa_completion_tokens += stats.get("generation_completion_tokens", 0)
                row_qa_times.append(stats.get("latency_seconds", 0))
                row_qa_tokens += stats.get("generation_tokens", 0)
                row_qa_prompt_tokens += stats.get("generation_prompt_tokens", 0)
                row_qa_completion_tokens += stats.get("generation_completion_tokens", 0)

            row_metrics = {
                "row_index": idx,
                "dataset_type": item.get("dataset_type"),
                "memory": {
                    "duration_seconds": mem_duration,
                    "prompt": mem_prompt_delta,
                    "completion": mem_completion_delta,
                    "total": mem_tokens_delta,
                    "calls": mem_calls_delta
                },
                "evolution": {
                    "duration_seconds": evo_duration,
                    "prompt": evo_prompt_delta,
                    "completion": evo_completion_delta,
                    "total": evo_tokens_delta,
                    "calls": evo_calls_delta
                },
                "qa": {
                    "question_count": len(row_qa_times),
                    "latency_seconds_list": row_qa_times,
                    "prompt": row_qa_prompt_tokens,
                    "completion": row_qa_completion_tokens,
                    "total": row_qa_tokens
                }
            }
            save_row_metrics_log(row_metrics, current_row_storage_path)

        finally:
            pipeline.shutdown_databases()

    # Aggregate accuracy and resource statistics.
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

    # 4.2 打印分类表格
    sorted_cats = sorted(stats_by_category.keys(), key=lambda x: str(x))
    for cat in sorted_cats:
        s = stats_by_category[cat]
        acc = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0.0

        display_name = str(cat)
        # Try to map friendly names for LoCoMo
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
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"详细结果Json已保存到: {output_file}")

    # Save performance report
    perf_file = os.path.join(TEST_ROOT_DIR, f"eval_performance_{DATASET_TYPE}.json")
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
    with open(perf_file, 'w', encoding='utf-8') as f:
        json.dump(perf_log, f, indent=2)
    print(f"性能统计日志已保存到: {perf_file}")

    print(f"中间数据库及详细Trace日志已保留在目录: {TEST_ROOT_DIR}")


if __name__ == '__main__':
    asyncio.run(main_async())
