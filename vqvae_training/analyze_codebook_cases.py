import json
from pathlib import Path
import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict, Counter
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.manifold import TSNE
from adjustText import adjust_text

# Use the shared training/inference architecture.
from src.utils.vqvae_t5_v10 import VQVAE_T5
from vqvae_training.train_vqvae import DailyDialogDataset, collate_fn

class IntegratedAnalyzer:
    def __init__(self, model_path, device='cuda', t5_model_name='t5-base'):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=False)
        self.model = VQVAE_T5(t5_model_name=t5_model_name, num_codebook_embeddings=1024, commitment_cost=0.10, decay=0.99)
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

        # 全局学术样式设置：强制加粗并设定大字号基础
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'axes.linewidth': 2.0,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'figure.dpi': 300
        })

    def run_full_analysis(self, dataloader, max_batches=200):
        code_to_tokens, code_frequency = defaultdict(list), Counter()
        continuous_features_list, assigned_codes_list = [], []

        print("Running unified inference...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Analyzing")):
                if i >= max_batches: break
                input_ids, att_mask = batch[0].to(self.device), batch[1].to(self.device)
                # 使用 Stride=1 的编码器提取连续特征
                outputs = self.model(input_ids, att_mask, input_ids, att_mask, input_ids)

                indices_cpu = outputs[4].cpu()
                flat_inputs = outputs[5] # 量化前的潜向量
                mask_cpu = att_mask.cpu()

                valid_mask = mask_cpu.view(-1).bool()
                continuous_features_list.append(flat_inputs[valid_mask].cpu().numpy())
                assigned_codes_list.append(indices_cpu.view(-1)[valid_mask].numpy())

                ids_cpu = input_ids.cpu()
                for b in range(ids_cpu.size(0)):
                    for t in range(ids_cpu.size(1)):
                        if mask_cpu[b, t] == 1:
                            idx = indices_cpu[b, t].item()
                            token = self.tokenizer.decode([ids_cpu[b, t].item()], skip_special_tokens=True)
                            if token.strip():
                                code_to_tokens[idx].append(token)
                                code_frequency[idx] += 1

        return (code_to_tokens, code_frequency,
                np.concatenate(continuous_features_list, axis=0),
                np.concatenate(assigned_codes_list, axis=0))

    def plot_composite_figure(self, code_to_tokens, code_frequency, cont_feats, assigned_codes, output_dir='./results'):
        os.makedirs(output_dir, exist_ok=True)

        # 1. 语义显著性计算
        global_counts = Counter()
        for tks in code_to_tokens.values(): global_counts.update(tks)
        total_tks = sum(global_counts.values())

        detailed_report = []
        for cid, count in code_frequency.most_common(20):
            tks = code_to_tokens[cid]
            c_counter = Counter(tks)
            scored = []
            for w, c_in in c_counter.items():
                if c_in < 3: continue
                lift = (c_in/len(tks)) / (global_counts[w]/total_tks + 1e-9)
                scored.append((w, lift * math.log(c_in + 1)))
            scored.sort(key=lambda x: x[1], reverse=True)
            detailed_report.append({"code_id": cid, "top_semantics": scored[:10]})

        # 2. 布局初始化：精细比例控制
        fig = plt.figure(figsize=(26, 22))
        # 设置 width_ratios [1, 1, 0.04]，第三列专门给 Heatmap 的 cbar
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.85], width_ratios=[1, 1, 0.04], hspace=0.25, wspace=0.2)

        ax_cont = fig.add_subplot(gs[0, 0])
        ax_disc = fig.add_subplot(gs[0, 1])
        ax_heat = fig.add_subplot(gs[1, 0:2]) # 占据第一二列实现完美对齐
        ax_cbar = fig.add_subplot(gs[1, 2])    # 独立放置色条

        sample_idx = np.random.choice(len(cont_feats), min(4500, len(cont_feats)), replace=False)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')

        emb_cont = tsne.fit_transform(cont_feats[sample_idx])
        all_weights = self.model.vq_layer.codebook.embedding.weight.detach().cpu().numpy()
        emb_disc = tsne.fit_transform(all_weights[assigned_codes[sample_idx]])

        jitter = (np.max(emb_disc) - np.min(emb_disc)) * 0.012
        emb_disc_j = emb_disc + np.random.normal(0, jitter, emb_disc.shape)

        top_12 = detailed_report[:12]
        target_ids = {e['code_id']: i for i, e in enumerate(top_12)}
        cmap = plt.cm.get_cmap('tab20', 12)

        # 绘制背景：降低可见度
        ax_cont.scatter(emb_cont[:, 0], emb_cont[:, 1], c='#DDDDDD', alpha=0.12, s=15)
        ax_disc.scatter(emb_disc_j[:, 0], emb_disc_j[:, 1], c='#DDDDDD', alpha=0.12, s=15)

        texts = []
        for i, cid in enumerate(assigned_codes[sample_idx]):
            if cid in target_ids:
                c_idx = target_ids[cid]
                # 散点大小适中，体现密度
                ax_cont.scatter(emb_cont[i,0], emb_cont[i,1], color=cmap(c_idx), s=40, alpha=0.75, edgecolors='none')
                ax_disc.scatter(emb_disc_j[i,0], emb_disc_j[i,1], color=cmap(c_idx), s=40, alpha=0.75, edgecolors='none')

        for ax, data in zip([ax_cont, ax_disc], [emb_cont, emb_disc_j]):
            x_min, x_max = np.min(data[:,0]), np.max(data[:,0])
            y_min, y_max = np.min(data[:,1]), np.max(data[:,1])
            m_x, m_y = (x_max - x_min)*0.1, (y_max - y_min)*0.1
            ax.set_xlim(x_min - m_x, x_max + m_x)
            ax.set_ylim(y_min - m_y, y_max + m_y)

        for i, entry in enumerate(top_12):
            cid = entry['code_id']
            kw = ", ".join([x[0] for x in entry['top_semantics'][:2]])
            mask = (assigned_codes[sample_idx] == cid)
            if np.any(mask):
                cx, cy = np.median(emb_disc_j[mask, 0]), np.median(emb_disc_j[mask, 1])
                t = ax_disc.text(cx, cy, f"[{cid}] {kw}", fontsize=14, weight='bold',
                                bbox=dict(boxstyle="round4,pad=0.3", fc="white", ec=cmap(i), lw=2, alpha=0.9))
                texts.append(t)
        adjust_text(texts, ax=ax_disc, arrowprops=dict(arrowstyle="->", color='#555555', lw=1.5, alpha=0.7))

        # 4. Heatmap 绘制逻辑 (大幅增强标注)
        heat_codes = detailed_report[:10]
        heat_kws = []
        for e in heat_codes: heat_kws.extend([x[0] for x in e['top_semantics'][:3]])
        unique_kws = sorted(list(set(heat_kws)), key=heat_kws.index)

        h_matrix = np.zeros((len(heat_codes), len(unique_kws)))
        for i, e in enumerate(heat_codes):
            s_map = {w: s for w, s in e['top_semantics']}
            for j, word in enumerate(unique_kws): h_matrix[i, j] = s_map.get(word, 0.0)

        # 传递特定的 ax_cbar 确保主图居中
        sns.heatmap(h_matrix, xticklabels=unique_kws, yticklabels=[f"Code {e['code_id']}" for e in heat_codes],
                    cmap="Blues", linewidths=1.5, ax=ax_heat, cbar_ax=ax_cbar,
                    cbar_kws={'label': 'Significance Lift Score'})

        # 坐标轴标注：大字号 + 加粗
        ax_heat.set_xticklabels(unique_kws, rotation=35, ha='right', weight='bold', fontsize=20)
        ax_heat.set_yticklabels([f"Code {e['code_id']}" for e in heat_codes], weight='bold', fontsize=20)

        ax_cbar.set_ylabel('Significance Lift Score', fontsize=22, weight='bold', labelpad=15)
        ax_cbar.tick_params(labelsize=16)
        for label in ax_cbar.get_yticklabels(): label.set_weight('bold')

        ax_cont.set_title("(a) Continuous Latent Manifold", fontsize=26, weight='bold', pad=25)
        ax_disc.set_title("(b) Discrete Semantic Clusters", fontsize=26, weight='bold', pad=25)
        ax_heat.set_title("(c) Multi-Cluster Semantic Decoupling Matrix", fontsize=26, weight='bold', pad=35)

        for ax in [ax_cont, ax_disc]:
            ax.grid(True, linestyle='--', alpha=0.4, lw=1)
            ax.tick_params(labelsize=18, width=2.5, length=6)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_weight('bold')

        # 导出 PDF 和 PNG
        plt.savefig(os.path.join(output_dir, 'Final_Academic_Integrated_Analysis.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'Final_Academic_Integrated_Analysis.png'), dpi=300, bbox_inches='tight')
        print(f"✅ 绘制成功！输出目录: {output_dir}")

def main():
    from settings import VQ_WEIGHTS_PATH, T5_MODEL_PATH
    CHECKPOINT = str(VQ_WEIGHTS_PATH)
    # 加载验证集
    val_loader = DataLoader(DailyDialogDataset(json.loads((Path(__file__).resolve().parents[1] / 'data/dailydialog/validation.json').read_text(encoding='utf-8')), T5Tokenizer.from_pretrained(str(T5_MODEL_PATH))),
                            batch_size=32, collate_fn=collate_fn)

    analyzer = IntegratedAnalyzer(model_path=CHECKPOINT, t5_model_name=str(T5_MODEL_PATH))
    c_map, c_freq, c_feats, a_codes = analyzer.run_full_analysis(val_loader, max_batches=200)
    analyzer.plot_composite_figure(c_map, c_freq, c_feats, a_codes)

if __name__ == "__main__":
    main()
