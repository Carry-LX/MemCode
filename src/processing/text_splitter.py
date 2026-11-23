'''
- 主要方法 `split_turn_messages` 接受一个代表单轮对话的消息列表（通常包含用户和助手两条消息）以及元数据。
- 根据参数 `split_into_sentences`，要么使用 `nltk.sent_tokenize` 把每条发言拆成句子并逐句产出，要么把整条发言作为一个单元产出。
- 每个产出项为字典，包含 `text`、`speaker`、`timestamp` 和 `source_id`。
- 文件 `src/processing/text_splitter.py` 底部包含一个示例/自测的 `__main__` 演示。
'''
import json
import nltk
from typing import List, Dict, Any, Generator

try:
    import nltk
    # 首次使用前，请确保已下载punkt分词器

except ImportError:
    print("错误: NLTK 库未安装。请运行 'pip install nltk' 并下载 'punkt' 数据。")
    nltk = None

class ConversationSplitter:
    """
    一个用于解析特定对话数据格式（如 longmemeval）并提取独立话语单元的工具。
    """

    def __init__(self):
        """
        初始化对话分割器。
        """
        if nltk is None:
            raise ImportError("NLTK 库未能导入，无法进行句子分割。")
        try:
            # 检查punkt是否可用，如果不可用会抛出异常
            nltk.data.find('tokenizers/punkt_tab/english/')
        except LookupError:
            raise RuntimeError("Install tokenizer resources before running: python -m nltk.downloader punkt punkt_tab")

    def split_turn_messages(
            self,
            turn_messages: List[Dict[str, Any]],
            timestamp: str,
            source_id: str,
            split_into_sentences: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        从单轮对话的消息中提取所有话语，并作为生成器逐一产生。

        Args:
            turn_messages (List[Dict[str, Any]]): 包含单轮对话中各条消息的列表。
                例如: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
            timestamp (str): 该轮对话的时间戳。
            source_id (str): 该轮对话所属的源ID（问题ID）。
            split_into_sentences (bool): 是否将每个话语内容分割成句子。

        Yields:
            一个包含话语信息的字典:
            {'text': str, 'speaker': str, 'timestamp': str, 'source_id': str}
        """
        # 依次处理该轮次中的 user 和 assistant 消息
        for message in turn_messages:
            role = message.get("role")
            content = message.get("content")

            if not role or not content:
                continue

            # 产生话语单元（分割或不分割）
            if split_into_sentences:
                sentences = nltk.sent_tokenize(content)
                for sentence in sentences:
                    yield {
                        "text": sentence.strip(),
                        "speaker": role,
                        "timestamp": timestamp,
                        "source_id": source_id,
                    }
            else:
                yield {
                    "text": content.strip(),
                    "speaker": role,
                    "timestamp": timestamp,
                    "source_id": source_id,
                }

# 第三步：内置的测试函数
if __name__ == '__main__':
    print("--- [Test] 单轮对话分割器 (ConversationSplitter) 功能演示 ---")

    test_turn = [
        {'role': 'user', 'content': "Can you solve a puzzle? It involves a fox and a chicken."},
        {'role': 'assistant', 'content': 'Of course! I love puzzles. Please tell me more.'}
    ]
    test_timestamp = "2024-05-22T14:00:00Z"
    test_source_id = "riddle_001"

    try:
        splitter = ConversationSplitter()

        print("\n[Test Scenario] 分割一个包含两条消息的对话轮...")
        print("=" * 60)

        # 将生成器结果转换为列表以便检查
        all_utterances = list(splitter.split_turn_messages(
            turn_messages=test_turn,
            timestamp=test_timestamp,
            source_id=test_source_id,
            split_into_sentences=True
        ))

        print(f"分割器总共提取了 {len(all_utterances)} 个话语单元（句子）。")
        for utt in all_utterances:
            print(f"  - {utt}")

        # 预期结果：User 2句 + Assistant 3句 = 5句
        expected_count = 5
        assert len(all_utterances) == expected_count, f"预期数量 {expected_count}, 实际 {len(all_utterances)}"
        assert all_utterances[0]['speaker'] == 'user'
        assert all_utterances[2]['speaker'] == 'assistant'
        assert all_utterances[0]['text'] == "Can you solve a puzzle?"
        assert all_utterances[-1]['text'] == "Please tell me more."
        assert all_utterances[0]['timestamp'] == test_timestamp

        print("\n[成功] 单轮分割器按预期工作。")

    except Exception as e:
        print(f"\n[错误] 测试无法运行: {e}")
        raise e

    print("\n--- [Test] 演示完成 ---")
