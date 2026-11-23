import torch
import torch.nn.functional as F
from transformers import T5Tokenizer
import os
from pathlib import Path

from src.utils.vqvae_t5_v10 import VQVAE_T5


class ModelInference:
    def __init__(self,
                 weights_path=None,  # Defaults are configured in settings.py.
                 t5_local_path=None,
                 device_str='cuda',
                 commitment_cost=0.10):

        print("\n=== Initializing VQ-VAE semantic encoder ===")

        from settings import VQ_WEIGHTS_PATH, T5_MODEL_PATH
        weights_path = Path(weights_path or VQ_WEIGHTS_PATH).expanduser().resolve()
        t5_local_path = str(t5_local_path or T5_MODEL_PATH)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Trained VQ-VAE checkpoint required: {weights_path}")

        # 2. 设备配置
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("⚠️ 使用 CPU")

        # 3. 加载 Tokenizer
        if os.path.exists(t5_local_path):
            self.tokenizer = T5Tokenizer.from_pretrained(t5_local_path, legacy=False)
        else:
            print(f"⚠️ 本地路径 {t5_local_path} 不存在，尝试使用 huggingface 默认路径")
            self.tokenizer = T5Tokenizer.from_pretrained(t5_local_path, legacy=False)

        # 4. 初始化模型结构
        print(f"Initializing architecture (stride=1, cost={commitment_cost})...")
        self.model = VQVAE_T5(
            t5_model_name=t5_local_path,
            t5_model_dir=t5_local_path if os.path.exists(t5_local_path) else None,
            num_codebook_embeddings=1024,
            commitment_cost=commitment_cost,
            decay=0.99
        )

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

    def get_indices(self, texts, max_len=64):
        """
        批量获取文本的语义哈希索引。

        Args:
            texts (str or List[str]): 输入文本或文本列表
            max_len (int): 最大序列长度，默认 64

        Returns:
            List[List[int]]: 嵌套列表，每个元素是去重后的索引列表 (保持序列顺序)
        """
        # 1. 输入预处理
        if isinstance(texts, str):
            texts = [texts]

        # 2. Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_len
        ).to(self.device)

        batch_results = []

        with torch.no_grad():
            # 3. Encoder Forward
            encoder_outputs = self.model.t5.encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            hidden_states = encoder_outputs.last_hidden_state  # [Batch, Seq, Dim]

            # 4. Compression (Stride=1)
            # [Batch, Seq, Dim] -> [Batch, Dim, Seq]
            hidden_states_transposed = hidden_states.permute(0, 2, 1)
            compressed = self.model.encoder_compress[0](hidden_states_transposed)
            # [Batch, Dim, Seq] -> [Batch, Seq, Dim]
            compressed = compressed.permute(0, 2, 1)
            compressed = F.gelu(self.model.compress_norm(compressed))

            # 5. VQ Quantization
            _, _, encoding_indices, _ = self.model.vq_layer(compressed)

            # 6. 后处理 (CPU)
            raw_indices_batch = encoding_indices.cpu().tolist()
            attention_mask_batch = inputs.attention_mask.cpu().tolist()

            for i in range(len(texts)):
                raw_indices = raw_indices_batch[i]
                mask = attention_mask_batch[i]

                # Step A: 过滤 Padding (只保留 mask=1 的部分)
                valid_indices = [idx for idx, m in zip(raw_indices, mask) if m == 1]

                # Step B: 连续去重 (Sequential Deduplication)
                # 逻辑: [A, A, B, A] -> [A, B, A] (保留顺序)
                dedup_indices = []
                if valid_indices:
                    dedup_indices.append(valid_indices[0])
                    for code in valid_indices[1:]:
                        if code != dedup_indices[-1]:
                            dedup_indices.append(code)

                batch_results.append(dedup_indices)

        return batch_results

    def get_hash_set(self, text):
        """Return unordered code-ID sets for set-based similarity calculations."""
        indices_list = self.get_indices(text)[0]
        return set(indices_list)


# ================================================================
# 测试入口 (Integration Test)
# ================================================================
if __name__ == "__main__":
    # Use the checkpoint and tokenizer configured in settings.py.


    from settings import VQ_WEIGHTS_PATH, T5_MODEL_PATH, DEVICE, root_path
    WEIGHTS_FILE = str(root_path(VQ_WEIGHTS_PATH))
    LOCAL_T5_DIR = str(T5_MODEL_PATH)

    print(f"测试权重路径: {os.path.abspath(WEIGHTS_FILE)}")

    if os.path.exists(WEIGHTS_FILE):
        # 初始化
        inference = ModelInference(
            weights_path=WEIGHTS_FILE,
            t5_local_path=LOCAL_T5_DIR,
            device_str=DEVICE,
            commitment_cost=0.10
        )

        # 1. 单句测试
        print("\n--- 单句测试 ---")
        single_text = "I need to fix my laptop screen."
        result = inference.get_indices(single_text)
        print(f"输入: {single_text}")
        print(f"哈希 (List): {result[0]}")

        # 2. 批量语义一致性测试
        print("\n--- 批量语义测试 ---")
        batch_texts = [
            "The weather is very cold today.",  # A
            "It is freezing outside right now.",  # B (应与 A 相似)
            "I want to buy an apple.",  # C (无关)
            "Stocks plunged due to uncertainty."  # D (无关)
        ]

        results = inference.get_indices(batch_texts)

        for text, indices in zip(batch_texts, results):
            print(f"\nText: {text}")
            print(f"Hash: {indices}")

    else:
        print(f"❌ 错误：找不到权重文件 {WEIGHTS_FILE}")
        print("请修改代码中的 WEIGHTS_FILE 变量为你实际的 .pt 文件路径。")
