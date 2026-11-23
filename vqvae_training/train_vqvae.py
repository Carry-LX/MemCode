import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import csv
import time
from datetime import datetime

# Use the shared training/inference architecture.
from src.utils.vqvae_t5_v10 import VQVAE_T5

# ==============================================================================
# 日志工具类
# ==============================================================================
class TrainingMonitor:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, 'training_metrics.csv')
        self.text_log_path = os.path.join(log_dir, 'generation_samples.txt')

        # 初始化 CSV 表头
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'epoch', 'step', 'global_step',
                    'total_loss', 'rec_loss', 'vq_loss',
                    'batch_unique_codes', 'active_codes_ratio',
                    'lr_t5', 'lr_vq'
                ])

    def log_metrics(self, metrics_dict):
        """记录数值指标到 CSV"""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%H:%M:%S"),
                metrics_dict['epoch'],
                metrics_dict['step'],
                metrics_dict['global_step'],
                f"{metrics_dict['total_loss']:.4f}",
                f"{metrics_dict['rec_loss']:.4f}",
                f"{metrics_dict['vq_loss']:.4f}",
                metrics_dict['batch_unique_codes'],
                f"{metrics_dict['active_codes_ratio']:.2f}",
                f"{metrics_dict['lr_t5']:.2e}",
                f"{metrics_dict['lr_vq']:.2e}"
            ])

    def log_text_sample(self, input_text, target_text, generated_text, step):
        """记录文本生成样本"""
        with open(self.text_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*20} Step {step} {'='*20}\n")
            f.write(f"Source: {input_text}\n")
            f.write(f"Target: {target_text}\n")
            f.write(f"Model : {generated_text}\n")

# ==============================================================================
# 数据集定义
# ==============================================================================
class DailyDialogDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_source_len=128, max_target_len=128, context_turns=2):
        self.data = []
        self.tokenizer = tokenizer
        for dialog in tqdm(dataset, desc="Processing Data"):
            utterances = dialog.get('utterances', dialog.get('dialog'))
            if not isinstance(utterances, list):
                raise ValueError('DailyDialog requires utterances/dialog as a list of strings')
            n = len(utterances)
            for i in range(context_turns, n):
                context_utt = utterances[i - context_turns:i]
                context = " </s> ".join(context_utt)
                response = utterances[i]
                if len(context.strip()) > 0 and len(response.strip()) > 0:
                    self.data.append({'context': context, 'response': response})
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        source = self.tokenizer(item['context'], max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        target = self.tokenizer(item['response'], max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': source.input_ids.squeeze(0), 'attention_mask': source.attention_mask.squeeze(0),
            'target_ids': target.input_ids.squeeze(0), 'target_attention_mask': target.attention_mask.squeeze(0)
        }

def collate_fn(batch):
    return (
        torch.stack([b['input_ids'] for b in batch]), torch.stack([b['attention_mask'] for b in batch]),
        torch.stack([b['target_ids'] for b in batch]), torch.stack([b['target_attention_mask'] for b in batch])
    )

# ==============================================================================
# 训练循环 (集成监控)
# ==============================================================================
def train_loop(model, tokenizer, train_loader, val_loader, device, epochs=10, base_lr=1e-3, t5_lr=1e-4, grad_accum_steps=2, save_dir='./checkpoints_v10', log_dir='./logs_v10'):

    monitor = TrainingMonitor(log_dir)

    # 参数分组：T5 vs VQ
    t5_param_ids = set(map(id, model.t5.parameters()))
    vq_params = [p for p in model.parameters() if id(p) not in t5_param_ids]
    t5_params = [p for p in model.parameters() if id(p) in t5_param_ids]

    optimizer = torch.optim.AdamW([
        {'params': t5_params, 'lr': t5_lr},    # Group 0
        {'params': vq_params, 'lr': base_lr}   # Group 1
    ], weight_decay=0.01)

    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    os.makedirs(save_dir, exist_ok=True)

    # Phase 0: 显式初始化 Codebook
    if model.vq_layer.codebook.initialized.item() == 0:
        print("\n=== Phase 0: 初始化 Codebook (防止坍缩) ===")
        model.eval()
        init_buffer = []
        collected = 0
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Calibration"):
                input_ids = batch[0].to(device)
                att_mask = batch[1].to(device)
                encoder_outputs = model.t5.encoder(input_ids=input_ids, attention_mask=att_mask)
                # Compress logic
                hidden = encoder_outputs.last_hidden_state
                compressed = model.encoder_compress[0](hidden.permute(0,2,1)).permute(0,2,1)
                compressed = F.gelu(model.compress_norm(compressed))

                # Filter padding
                mask_bool = att_mask.bool().unsqueeze(-1)
                valid = torch.masked_select(compressed, mask_bool).view(-1, compressed.size(-1))
                init_buffer.append(valid)
                collected += valid.size(0)
                if collected >= 4096: break

        model.vq_layer.codebook.init_codebook(torch.cat(init_buffer, dim=0))
        print("✅ Codebook 初始化完成！\n")

    # Phase 1: 训练
    global_step = 0
    model.train()

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1} / {epochs} ===")
        progress_bar = tqdm(train_loader, desc=f"Train")

        for step, (input_ids, att_mask, target_ids, target_mask) in enumerate(progress_bar):
            input_ids, att_mask = input_ids.to(device), att_mask.to(device)
            target_ids, target_mask = target_ids.to(device), target_mask.to(device)

            labels = target_ids[:, 1:].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # Forward
            loss, rec_loss, vq_loss, logits, encoding_indices, flat_inputs = model(
                input_ids, att_mask, target_ids[:, :-1], target_mask[:, :-1], labels
            )

            loss = loss / grad_accum_steps
            loss.backward()

            # 指标计算
            with torch.no_grad():
                mask_bool = att_mask.bool()
                valid_indices = torch.masked_select(encoding_indices, mask_bool)
                batch_unique = torch.unique(valid_indices).numel()
                # 估算当前 batch 激活的 code 比例 (0.0 - 1.0)
                active_ratio = batch_unique / 1024.0

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 死码复活 (每 20 步)
                if global_step % 20 == 0:
                    mask_flat = att_mask.view(-1).bool()
                    model.vq_layer.codebook.revive_dead_codes(flat_inputs[mask_flat].detach())

                # 核心：每 10 步记录一次 CSV 日志
                if global_step % 10 == 0:
                    monitor.log_metrics({
                        'epoch': epoch + 1,
                        'step': step,
                        'global_step': global_step,
                        'total_loss': loss.item() * grad_accum_steps,
                        'rec_loss': rec_loss.item(),
                        'vq_loss': vq_loss.item(),
                        'batch_unique_codes': batch_unique,
                        'active_codes_ratio': active_ratio,
                        'lr_t5': optimizer.param_groups[0]['lr'],
                        'lr_vq': optimizer.param_groups[1]['lr']
                    })

                # 核心：每 200 步记录一次文本生成采样
                if global_step % 200 == 0:
                    # 简单取 batch 里第一条数据看看生成效果
                    pred_ids = torch.argmax(logits[0], dim=-1)
                    src_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    tgt_text = tokenizer.decode(target_ids[0], skip_special_tokens=True)
                    gen_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                    monitor.log_text_sample(src_text, tgt_text, gen_text, global_step)

            progress_bar.set_postfix({
                'L': f"{loss.item()*grad_accum_steps:.3f}",
                'Uniq': batch_unique
            })

        torch.save(model.state_dict(), os.path.join(save_dir, f'vqvae_t5_v10_epoch{epoch+1}.pt'))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Explicit model and local dataset paths.
    from settings import T5_MODEL_PATH
    t5_path = str(T5_MODEL_PATH)

    tokenizer = T5Tokenizer.from_pretrained(t5_path, legacy=False)
    import json
    from pathlib import Path
    training_dir = Path(__file__).resolve().parents[1] / "data" / "dailydialog"
    dataset = {split: json.loads((training_dir / f"{split}.json").read_text(encoding="utf-8"))
               for split in ("train", "validation")}


    train_loader = DataLoader(
        DailyDialogDataset(dataset['train'], tokenizer),
        batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        DailyDialogDataset(dataset['validation'], tokenizer),
        batch_size=16, collate_fn=collate_fn
    )

    model = VQVAE_T5(
        t5_model_name="t5-base",
        t5_model_dir=t5_path,
        num_codebook_embeddings=1024,
        commitment_cost=0.10,
        decay=0.99
    )
    model.to(device)

    # 启动训练
    train_loop(
        model, tokenizer, train_loader, val_loader, device,
        epochs=10,
        base_lr=1e-3, # VQ 层学习率
        t5_lr=1e-4,   # T5 学习率
        save_dir='./models/T5', log_dir='./runs/vqvae_training' # 日志将保存在这里
    )

if __name__ == "__main__":
    main()
