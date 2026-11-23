# src/utils/llm_interface.py
import time
import os
import re
from typing import Dict

try:
    from openai import OpenAI
except ImportError:
    print("错误: 'openai' 库未安装。请运行: pip install openai")
    OpenAI = None


class OpenAIInterface:
    """
    一个与 OpenAI API 交互的具体实现，增加了Token使用情况的追踪功能。
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini",base_url: str = "https://api.openai.com/v1"):
        if OpenAI is None:
            raise ImportError("OpenAI 库未能导入。")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key 未提供。请在初始化时传入，或设置 'OPENAI_API_KEY' 环境变量。")

        self.base_url =  base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
        print(f"当前使用模型: {self.model}")

        # 初始化Token计数器
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    def get_completion(self, prompt: str) -> str:
        """
        从 OpenAI 模型获取响应，并记录Token使用情况。
        """
        retry = 0  # 重试计数器
        max_retry = 20  # 最大重试20次

        # Retries use the original prompt unchanged.
        current_prompt = prompt

        while retry < max_retry:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant that provides responses in the format requested."},
                        {"role": "user", "content": current_prompt}
                    ],
                    temperature=0.0
                )

                # 捕获并累加Token使用量
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    # 打印单次调用的使用情况（可选，用于调试）


                return response.choices[0].message.content

            except Exception as e:
                retry += 1

                # Stop on provider policy rejection without changing the prompt.
                error_msg = str(e).lower()
                is_policy_error = "inappropriate content" in error_msg or \
                                  ("429" in error_msg and "bad_response" in error_msg)

                if is_policy_error:
                    raise RuntimeError("Provider rejected the request.") from e

                # 输出错误日志
                with open("error_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"[Error] Retry {retry} | Error: {e}\n\n")
                print(f"调用 OpenAI API 时发生错误 (重试{retry}/{max_retry}): {e}")

                # 重试20次失败则抛出错误
                if retry >= max_retry:
                    raise e
                # Wait one second between transient retries.
                time.sleep(1)
        return ""

    # 提供一个获取总使用量的方法
    def get_total_usage(self) -> Dict[str, int]:
        """
        返回累计的Token使用总量。
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }
