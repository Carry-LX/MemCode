import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
import os


class EuclideanCodebook(nn.Module):
    """Normalized codebook with EMA updates and data-driven initialization."""

    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # 均匀分布初始化并归一化
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)

        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.zeros(1))

    def init_codebook(self, x):
        # x shape: [N, Dim]
        print(f"Data-driven codebook initialization: {x.size(0)} input samples")

        # 简单去重，防止大量重复的 Padding 向量占据码本
        # 注意：这里只是为了初始化更好，不影响后续训练
        unique_x, _ = torch.unique(x, dim=0, return_inverse=True)
        if unique_x.size(0) < x.size(0):
            # 如果重复很多（说明有很多 Padding），我们尽量取 unique 的
            # 但如果 unique 的太少，还是得用原来的
            if unique_x.size(0) > self.num_embeddings // 2:
                x = unique_x

        if x.size(0) < self.num_embeddings:
            repeat_times = (self.num_embeddings // x.size(0)) + 1
            x = x.repeat(repeat_times, 1)

        perm = torch.randperm(x.size(0)).to(x.device)
        sampled = x[perm[:self.num_embeddings]]
        sampled = F.normalize(sampled, p=2, dim=-1)

        self.embedding.weight.data.copy_(sampled)
        self.embed_avg.data.copy_(sampled)
        self.cluster_size.data.fill_(1.0)
        self.initialized.fill_(1)

    def forward(self, x):
        # x 必须先归一化
        x_norm = F.normalize(x, p=2, dim=-1)

        # 训练阶段且未初始化时，执行初始化
        if self.training and self.initialized.item() == 0:
            self.init_codebook(x)

        w_norm = F.normalize(self.embedding.weight, p=2, dim=-1)

        distances = (
                torch.sum(x_norm ** 2, dim=1, keepdim=True) +
                torch.sum(w_norm ** 2, dim=1) -
                2 * torch.matmul(x_norm, w_norm.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)
        quantized = F.normalize(quantized, p=2, dim=-1)

        if self.training:
            encodings_one_hot = torch.zeros(
                encoding_indices.size(0), self.num_embeddings, device=x.device
            )
            encodings_one_hot.scatter_(1, encoding_indices.unsqueeze(1), 1)

            self.cluster_size.data.mul_(self.decay).add_(
                encodings_one_hot.sum(0), alpha=1 - self.decay
            )

            embed_sum = torch.matmul(encodings_one_hot.t(), x_norm)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size_smoothed = (
                    (self.cluster_size + self.epsilon) /
                    (n + self.num_embeddings * self.epsilon) * n
            )
            embed_normalized = self.embed_avg / cluster_size_smoothed.unsqueeze(1)
            self.embedding.weight.data.copy_(F.normalize(embed_normalized, p=2, dim=-1))

        return quantized, encoding_indices

    def revive_dead_codes(self, x):
        if not self.training: return 0
        threshold = 1.0
        dead_indices = (self.cluster_size < threshold).nonzero(as_tuple=True)[0]
        num_dead = len(dead_indices)
        if num_dead > 0:
            x_norm = F.normalize(x, p=2, dim=-1)
            # 随机扰动输入，防止复活成一样的值
            perm = torch.randperm(x_norm.size(0)).to(x.device)
            if x_norm.size(0) < num_dead:
                idx = perm.repeat((num_dead // x_norm.size(0)) + 1)[:num_dead]
            else:
                idx = perm[:num_dead]
            selected_inputs = x_norm[idx]
            # 增加一点噪声
            selected_inputs = selected_inputs + torch.randn_like(selected_inputs) * 0.02
            selected_inputs = F.normalize(selected_inputs, p=2, dim=-1)

            with torch.no_grad():
                self.embedding.weight.data[dead_indices] = selected_inputs
                self.embed_avg.data[dead_indices] = selected_inputs
                self.cluster_size.data[dead_indices] = 1.0
        return num_dead


class ImprovedVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.codebook = EuclideanCodebook(num_embeddings, embedding_dim, decay=decay)

    def compute_entropy_loss(self, encoding_indices):
        encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_usage = encodings_one_hot.mean(dim=0) + 1e-10
        entropy_loss = (avg_usage * torch.log(avg_usage)).sum()
        return entropy_loss

    def forward(self, x):
        x_flat = x.reshape(-1, self.embedding_dim)
        quantized_flat, encoding_indices = self.codebook(x_flat)
        quantized = quantized_flat.view(x.shape)

        x_norm = F.normalize(x, p=2, dim=-1)
        e_latent_loss = F.mse_loss(quantized.detach(), x_norm)
        vq_loss = self.commitment_cost * e_latent_loss

        if self.training:
            entropy_loss = self.compute_entropy_loss(encoding_indices)
            vq_loss += entropy_loss * 0.01

        quantized = x_norm + (quantized - x_norm).detach()
        return quantized, vq_loss, encoding_indices.view(x.shape[0], x.shape[1]), x_flat


class VQVAE_T5(nn.Module):
    def __init__(
            self,
            t5_model_name="t5-base",
            t5_model_dir=None,
            num_codebook_embeddings=1024,
            commitment_cost=0.1,
            decay=0.99,
            vq_loss_weight=1.0
    ):
        super().__init__()

        if t5_model_dir and os.path.exists(t5_model_dir):
            self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_dir)
        else:
            self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

        self.codebook_dim = self.t5.config.d_model
        self.t5_hidden_size = self.t5.config.d_model

        # Stride=1
        self.encoder_compress = nn.Sequential(
            nn.Conv1d(self.t5_hidden_size, self.codebook_dim, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(self.codebook_dim),
            nn.GELU()
        )

        self.vq_layer = ImprovedVectorQuantizer(
            num_embeddings=num_codebook_embeddings,
            embedding_dim=self.codebook_dim,
            commitment_cost=commitment_cost,
            decay=decay
        )

        self.decoder_decompress = nn.Sequential(
            nn.Conv1d(self.codebook_dim, self.t5_hidden_size, kernel_size=3, stride=1, padding=1)
        )

        self.post_vq_proj = nn.Linear(self.codebook_dim, self.t5_hidden_size)
        self.vq_loss_weight = vq_loss_weight
        self.compress_norm = nn.LayerNorm(self.codebook_dim)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        # Encoder
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        # Compress (Stride=1)
        hidden_states_transposed = hidden_states.permute(0, 2, 1)
        compressed_states = self.encoder_compress[0](hidden_states_transposed)
        compressed_states = compressed_states.permute(0, 2, 1)
        compressed_states = self.compress_norm(compressed_states)
        compressed_states = F.gelu(compressed_states)

        # VQ
        quantized, vq_loss, encoding_indices, flat_inputs = self.vq_layer(compressed_states)

        # Decompress (Stride=1)
        quantized_scaled = self.post_vq_proj(quantized)
        quantized_transposed = quantized_scaled.permute(0, 2, 1)
        reconstructed_states = self.decoder_decompress(quantized_transposed)
        reconstructed_states = reconstructed_states.permute(0, 2, 1)

        # Decoder
        decoder_outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=reconstructed_states,
            encoder_attention_mask=attention_mask,
            use_cache=False
        )
        lm_logits = self.t5.lm_head(decoder_outputs.last_hidden_state)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        reconstruction_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        total_loss = reconstruction_loss + self.vq_loss_weight * vq_loss

        return total_loss, reconstruction_loss, vq_loss, lm_logits, encoding_indices, flat_inputs

    def get_codebook_stats(self):
        return {'active_codes': (self.vq_layer.codebook.cluster_size > 0.1).sum().item()}
