import os
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_path: str, batch_size: int = 32, use_fp16: bool = False):
        """
        EmbeddingModel wrapper.

        Args:
            model_path: 模型路径或 HuggingFace Hub 名称
            batch_size: encode 时的 batch size
            use_fp16: 是否在推理时尝试启用混合精度（仅在 CUDA 可用时生效）
                      注意：默认 False（精度优先）。如果设为 True，会在 encode 时使用 torch.cuda.amp.autocast()
        """
        self.model = SentenceTransformer(model_path)
        self.batch_size = batch_size
        self.use_fp16 = use_fp16

        # 将模型搬到 CUDA（如果可用），但不对模型进行全局 .half() 转换（避免数值问题）
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        # 打印状态，便于调试
        print(f"模型 '{model_path}' 已加载。使用 GPU: {torch.cuda.is_available()}, use_fp16 flag: {use_fp16}")

    def _encode_with_optional_autocast(self, texts: List[str], **kwargs):
        """
        内部辅助：在需要时使用 autocast 包裹 encode 调用。
        SentenceTransformer.encode 返回 numpy array（如果 convert_to_tensor=False），
        我们之后会强制转为 float32。
        """
        if self.use_fp16 and torch.cuda.is_available():
            # 使用混合精度上下文（更稳健 than model.half()）
            with torch.cuda.amp.autocast():
                embeddings = self.model.encode(texts, **kwargs)
        else:
            embeddings = self.model.encode(texts, **kwargs)
        return embeddings

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入生成 - 返回 float32 列表"""
        if not texts:
            return []

        # encode 参数：保持 normalize_embeddings=True 以便使用 cosine 相似度 或 L2(单位向量) 兼容
        embeddings = self._encode_with_optional_autocast(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=True
        )

        # 确保 numpy array 并强制转换为 float32（避免 fp16 传播到后续存储）
        emb_np = np.asarray(embeddings)
        if emb_np.dtype != np.float32:
            emb_np = emb_np.astype(np.float32)

        return emb_np.tolist()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文本生成嵌入向量。保证返回 float32。
        """
        if not texts or not isinstance(texts, list):
            print("警告: 输入的 texts 为空或格式不正确，返回空列表。")
            return []

        embeddings = self._encode_with_optional_autocast(
            texts,
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=True
        )

        emb_np = np.asarray(embeddings)
        if emb_np.dtype != np.float32:
            emb_np = emb_np.astype(np.float32)

        return emb_np.tolist()


# 第三步：内置的测试函数
if __name__ == '__main__':
    print("--- [Test] 嵌入模型 (EmbeddingModel) 功能演示 ---")

    MODEL_PATH = "../../models/all-MiniLM-L6-v2"

    try:
        print(f"\n[Step 1] 尝试初始化 EmbeddingModel...")
        # 默认 use_fp16=False，保证精度
        embedder = EmbeddingModel(model_path=MODEL_PATH, use_fp16=False)

        print("\n[Step 2] 准备示例文本进行嵌入...")
        sample_texts = [
            "项目Alpha的最终预算被确定为15万元。",
            "这是一个完全无关的句子。",
            "会议讨论并确定了项目Alpha的预算及后续步骤。"
        ]
        for i, text in enumerate(sample_texts):
            print(f"  {i + 1}. {text}")

        print("\n[Step 3] 调用 embed() 方法生成向量...")
        vectors = embedder.embed(sample_texts)

        print("\n[Step 4] 验证输出结果...")
        if vectors:
            import numpy as np
            arr = np.asarray(vectors)
            print(f"成功生成了 {len(vectors)} 个向量。")
            print(f"每个向量的维度是: {arr.shape[1]}")
            print(f"向量 dtype: {arr.dtype}")
            print("第一个句子的向量 (部分展示):")
            print(vectors[0][:10])
        else:
            print("未能生成向量。")

    except (IOError, ImportError) as e:
        print(f"\n[错误] 测试失败: {e}")
        print("\n请确认以下事项：")
        print(f"1. 模型文件夹 '{MODEL_PATH}' 是否存在。")
        print("2. 是否已通过 'pip install sentence-transformers' 安装了必要的库。")

    print("\n--- [Test] 演示完成 ---")
