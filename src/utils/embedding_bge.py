from sentence_transformers import SentenceTransformer
from typing import List, Union


class EmbeddingModel:
    def __init__(self, model_path: str = "BAAI/bge-base-en-v1.5", device: str = "cuda"):
        print(f"正在加载 BGE 嵌入模型: {model_path} ...")
        self.model = SentenceTransformer(model_path, device=device)

        # 定义 BGE 专属的查询前缀
        self.query_instruction = "Represent this sentence for searching relevant passages: "

    def embed(self, texts: Union[str, List[str]], is_query: bool = False) -> List[List[float]]:
        """
        生成嵌入向量。
        Args:
            texts: 文本列表或单条文本
            is_query: 关键参数！如果是搜问题，设为 True；如果是存事实，设为 False。
        """
        # 统一转为列表处理
        if isinstance(texts, str):
            texts = [texts]


        if is_query:
            # 如果是查询，给每个文本加上前缀
            # 例如: "Who is Melanie?" -> "Represent this sentence...: Who is Melanie?"
            processed_texts = [f"{self.query_instruction}{t}" for t in texts]
            # 打印一下，方便你调试确认

        else:
            # 如果是原子事实（文档），不需要加前缀，直接用原文本
            processed_texts = texts


        # BGE 建议开启 normalize_embeddings=True，这对余弦相似度检索至关重要
        embeddings = self.model.encode(processed_texts, normalize_embeddings=True)

        # 确保返回 list 格式
        return embeddings.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量处理原子事实入库
        注意：这里默认 is_query=False，因为这是在存数据库，不是在查
        """
        return self.embed(texts, is_query=False)
