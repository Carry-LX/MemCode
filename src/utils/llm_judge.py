
from typing import Any, Optional

class LLMJudge:
    """
    一个使用 LLM 作为裁判的评估器，根据预设的Prompt模板来判断模型答案的正确性。
    """

    # 使用类变量统一定义类别，避免硬编码字符串
    STANDARD_TASK = "standard"
    TEMPORAL_REASONING = "temporal_reasoning"
    KNOWLEDGE_UPDATE = "knowledge_update"
    PREFERENCE_TASK = "preference"
    ABSTENTION_TASK = "abstention"

    def __init__(self, llm_interface: Any):
        """
        初始化 LLM 裁判。

        Args:
            llm_interface (Any): 一个实现了 get_completion(prompt) 方法的对象实例，
                                 例如 OpenAIInterface。
        """
        self.llm = llm_interface
        # 将类别映射到对应的prompt构建方法，使代码更清晰、可扩展
        self.prompt_builders = {
            self.STANDARD_TASK: self._build_standard_prompt,
            self.TEMPORAL_REASONING: self._build_temporal_prompt,
            self.KNOWLEDGE_UPDATE: self._build_knowledge_update_prompt,
            self.PREFERENCE_TASK: self._build_preference_prompt,
            self.ABSTENTION_TASK: self._build_abstention_prompt,
        }
        print("LLM 裁判 (LLMJudge) 已初始化。")

# Task-specific evaluator prompts. These are the project evaluation protocol.

    # 1. 通用构建方法：标准任务
    def _build_standard_prompt(self, question: str, correct_answer: str, model_response: str,
                               query_date: str = None, **kwargs) -> str:
        # 局部变量，使用 f-string 直接注入
        time_context = f"Reference Time: {query_date}\n" if query_date else ""

        # 注意：time_context 用 {}，因为它已经在上面计算好了
        # question 等参数用 {{}}，因为它们要留给后面的 .format() 处理
        return f"""I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.
                {time_context}Question: {{question}}
                Correct Answer: {{correct_answer}}
                Model Response: {{model_response}}
                Is the model response correct? Answer yes or no only.""".format(
            question=question, correct_answer=correct_answer, model_response=model_response
        )

    # Temporal evaluation, including duration counting conventions.
    def _build_temporal_prompt(self, question: str, correct_answer: str, model_response: str,
                               query_date: str = None, **kwargs) -> str:
        context_info_block = ""
        if query_date:
            context_info_block = f"\n                [Context Info]:\n                Reference Time: {query_date}\n"

        return f"""
                You are an intelligent judge evaluating the correctness of an AI's answer regarding time and dates.
                {context_info_block}
                [Question]: {{question}}

                [Ground Truth]: {{correct_answer}}

                [Model Answer]: {{model_response}}

                [Evaluation Rules - READ CAREFULLY]:
                1. **Date Format Equivalence**:
                   - "2025-05-07" == "7 May 2025" == "May 7th".

                2. **Semantic Inclusion**: If the Ground Truth is a simple time and the Model Answer contains that time in a full sentence, it is **CORRECT**.

                3. **Alternative Answers (CRITICAL)**:
                   - If the Ground Truth provides multiple options (e.g., "X or Y", "X. Y is also acceptable"), the Model Answer is CORRECT if it matches **ANY ONE** of the options.
                   - The model does NOT need to list all options.

                4. **Counting Ambiguity (Inclusive vs Exclusive)**:
                   - For duration questions (e.g., "How many days"), allow differences caused by counting logic.
                   - Example: "Mon to Wed" can be 2 days (Wed-Mon) or 3 days (Mon, Tue, Wed). **BOTH are considered CORRECT.**
                   - Explicitly allow +/- 1 unit difference for durations.

                5. **Reference Consistency**: If the model answers relative to the provided [Reference Time] (if any) correctly matching the Ground Truth logic, it is CORRECT.

                Based on these rules, does the [Model Answer] factually match the [Ground Truth] (or one of its valid options)?
                Respond with ONLY "yes" or "no".
                """.format(
            question=question, correct_answer=correct_answer, model_response=model_response
        )

    # 3. 知识更新
    def _build_knowledge_update_prompt(self, question: str, correct_answer: str, model_response: str,
                                       query_date: str = None, **kwargs) -> str:
        time_context = f"Reference Time: {query_date}\n" if query_date else ""

        return f"""I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.
                {time_context}Question: {{question}}
                Correct Answer: {{correct_answer}}
                Model Response: {{model_response}}
                Is the model response correct? Answer yes or no only.""".format(
            question=question, correct_answer=correct_answer, model_response=model_response
        )

    # 4. 偏好任务
    def _build_preference_prompt(self, question: str, rubric: str, model_response: str, query_date: str = None,
                                 **kwargs) -> str:
        time_context = f"Reference Time: {query_date}\n" if query_date else ""

        return f"""I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.
                {time_context}Question: {{question}}
                Rubric: {{rubric}}
                Model Response: {{model_response}}
                Is the model response correct? Answer yes or no only.""".format(
            question=question, rubric=rubric, model_response=model_response
        )

    # 5. 拒答任务
    def _build_abstention_prompt(self, question: str, explanation: str, model_response: str, query_date: str = None,
                                 **kwargs) -> str:
        time_context = f"Reference Time: {query_date}\n" if query_date else ""

        return f"""I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.
                {time_context}Question: {{question}}
                Explanation: {{explanation}}
                Model Response: {{model_response}}
                Does the model correctly identify the question as unanswerable? Answer yes or no only.""".format(
            question=question, explanation=explanation, model_response=model_response
        )

    # Evaluate an answer with the selected task-specific prompt.
    def evaluate(self,
                 category: str,
                 question: str,
                 model_response: str,
                 correct_answer: Optional[str] = None,
                 rubric: Optional[str] = None,
                 explanation: Optional[str] = None,
                 query_date: Optional[str] = None) -> bool:  # 默认为 None

        if category not in self.prompt_builders:
            raise ValueError(f"未知的评判类别: '{{category}}'")

        builder = self.prompt_builders[category]

        # 将 query_date 传给 builder
        # All task builders receive the optional query reference time.
        prompt = builder(
            question=question,
            correct_answer=correct_answer,
            model_response=model_response,
            rubric=rubric,
            explanation=explanation,
            query_date=query_date
        )

        # 调用 LLM 进行评判
        judge_response = self.llm.get_completion(prompt)

        if not judge_response:
            print("警告: 裁判LLM未返回任何评判结果。")
            return False

        # 解析裁判的回答
        return judge_response.strip().lower() == 'yes'


# 内置的完整测试函数
if __name__ == '__main__':
    print("--- [Test] LLM 裁判 (LLMJudge) 功能测试 ---")


    # 1. 创建一个 Mock LLM Interface 用于测试
    class MockLLMInterface:
        def get_completion(self, prompt: str) -> str:
            """一个可预测的模拟LLM，用于返回 'yes' 或 'no'。"""
            print("\n--- Mock 裁判 LLM 接收到的 Prompt ---")
            print(prompt)
            print("--------------------------------------")

            # 模拟裁判的逻辑
            if "response contains the correct answer" in prompt and "Paris" in prompt and "Correct Answer: Paris" in prompt:
                return "yes"
            if "previous information along with an updated answer" in prompt and "New York" in prompt and "Tokyo" in prompt:
                return "yes"  # 模拟接受包含旧信息的更新
            if "rubric for desired personalized response" in prompt and "blue" in prompt:
                return "yes"  # 模拟满足个性化偏好
            if "unanswerable" in prompt and "cannot determine" in prompt:
                return "yes"  # 模拟正确识别出无法回答

            return "no"


    # 2. 初始化裁判
    mock_llm = MockLLMInterface()
    judge = LLMJudge(llm_interface=mock_llm)

    # 3. 为每个类别创建测试场景并进行验证

    # 场景 1: 标准任务 (Standard Task)
    print("\n--- [场景 1] 标准任务 ---")
    is_correct = judge.evaluate(
        category=LLMJudge.STANDARD_TASK,
        question="What is the capital of France?",
        correct_answer="Paris",
        model_response="The capital city of France is Paris."
    )
    assert is_correct, "标准任务评判失败！"
    print("[成功] 标准任务评判正确。")

    # 场景 3: 知识更新任务 (Knowledge Update Task)
    print("\n--- [场景 3] 知识更新任务 ---")
    is_correct = judge.evaluate(
        category=LLMJudge.KNOWLEDGE_UPDATE,
        question="What is the user's current location? They just moved.",
        correct_answer="The user is now in Tokyo.",
        model_response="The user used to live in New York, but their current location is Tokyo."
    )
    assert is_correct, "知识更新任务评判失败！"
    print("[成功] 知识更新任务评判正确。")

    # 场景 4: 个性化偏好任务 (Preference Task)
    print("\n--- [场景 4] 个性化偏好任务 ---")
    is_correct = judge.evaluate(
        category=LLMJudge.PREFERENCE_TASK,
        question="Suggest a color for my new car.",
        rubric="The user has previously mentioned their favorite color is blue.",
        model_response="Considering your favorite color, a deep blue would be a great choice."
    )
    assert is_correct, "个性化偏好任务评判失败！"
    print("[成功] 个性化偏好任务评判正确。")

    # 场景 5: 拒绝回答任务 (Abstention Task)
    print("\n--- [场景 5] 拒绝回答任务 ---")
    is_correct = judge.evaluate(
        category=LLMJudge.ABSTENTION_TASK,
        question="What was the exact closing price of AAPL on Jan 1st, 2022?",
        explanation="The provided documents do not contain specific stock market data.",
        model_response="Based on the information I have, I cannot determine the exact closing price."
    )
    assert is_correct, "拒绝回答任务评判失败！"
    print("[成功] 拒绝回答任务评判正确。")

    print("\n--- [Test] 所有评判场景均已通过 ---")
