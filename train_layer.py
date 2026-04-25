import os
import argparse
import time
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

# 激活高精度张量核心计算
torch.set_float32_matmul_precision('high')

from LNNModel import FluxPropLanguageModel
from freeze_utils import freeze_all_layers_except

# ==========================================
# 硬件极致压榨配置
# ==========================================
SEQ_LEN = 256        
HIDDEN_DIM = 1024    
RANK = 128           
BATCH_SIZE = 384     # 配合连续流适当调整，防止显存爆炸
LEARNING_RATE = 6e-4 

torch.backends.cudnn.benchmark = True 
torch.manual_seed(42)

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]
    def decode(self, tokens):
        return ''.join([self.itos[tok] for tok in tokens])
    def vocab_size(self):
        return len(self.chars)

# 真正的连续状态流数据迭代器
class StatefulDataLoader:
    def __init__(self, data_tensor, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # 将一维数据折叠成 BATCH_SIZE 个平行的连续流
        n_seqs = len(data_tensor) // batch_size
        # 裁剪掉多余的尾巴，确保完美变形
        data_tensor = data_tensor[:n_seqs * batch_size].view(batch_size, n_seqs)
        self.data = data_tensor
        self.n_batches = (n_seqs - 1) // seq_len

    def __iter__(self):
        for i in range(self.n_batches):
            start_idx = i * self.seq_len
            end_idx = start_idx + self.seq_len
            x = self.data[:, start_idx : end_idx]
            y = self.data[:, start_idx + 1 : end_idx + 1]
            yield x, y
            
    def __len__(self):
        return self.n_batches

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="正在训练的层级索引 (0, 1, 2...)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("⏳ 正在加载语料并进行全局预编码...")
    with open("train_en.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer = CharTokenizer(raw_text)
    full_data_tensor = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    
    loader = StatefulDataLoader(full_data_tensor, BATCH_SIZE, SEQ_LEN)
    print(f"✅ 数据流构建完成: 每轮 Epoch 包含 {len(loader)} 个连续时间步批次。")

    print(f"📦 初始化 FluxProp 架构 (Dim={HIDDEN_DIM}, Rank={RANK})")
    model = FluxPropLanguageModel(
        vocab_size=tokenizer.vocab_size(),
        embed_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        num_layers=8, 
        device=device
    ).to(device)

    ckpt_dir = f"checkpoints/layer_{args.layer:02d}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if args.layer > 0:
        prev_ckpt = f"checkpoints/layer_{args.layer-1:02d}/final.pt"
        if os.path.exists(prev_ckpt):
            state_dict = torch.load(prev_ckpt)["model_state"]
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            # [FIX] 加载完毕后强制重建 weight tying，防止 lm_head 与 embedding 解绑
            model.lm_head.weight = model.embedding.weight
            print(f"✅ 已继承 Layer {args.layer-1} 的预训练知识 (weight tying 已重建)")
            
    # 必须先冻结物理层，然后再进行 compile，否则冻结不生效
    freeze_all_layers_except(model, args.layer)

    print("🔥 正在使用 torch.compile 对模型进行底层计算图融合优化...")
    #model = torch.compile(model)

    # 精细化的优化器策略，保护 LNN 动力学参数不被 Weight Decay 归零
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith('bias') or 'alpha' in name or 'gamma' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optim_groups = [
        {'params': decay, 'weight_decay': 0.01},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE)
    
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=f"runs/layer_{args.layer}")

    tokens_per_step = BATCH_SIZE * SEQ_LEN
    total_tokens_processed = 0

    print(f"🚀 ===================================================")
    print(f"🚀 开始满血训练 Layer {args.layer} | 序列: {SEQ_LEN} | 批次: {BATCH_SIZE}")
    print(f"🚀 ===================================================")
    
    step = 0
    model.train()
    last_log_time = time.time()
    
    # 维持跨批次状态的全局缓存
    h_states = None
    
    try:
        while True:
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                # 截断梯度，防止反向传播穿透到上一个 Batch，导致 OOM
                if h_states is not None:
                    h_states = [h.detach() for h in h_states]

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    logits, h_states = model(x, h_states)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                
                scaler.scale(loss).backward()
                # [FIX] 关键修复：必须先 unscale 再裁剪，否则 max_norm=1.0 是在数千倍放大后的梯度上裁剪，等于无效
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                step += 1
                total_tokens_processed += tokens_per_step
                
                writer.add_scalar("Training/Loss", loss.item(), step)
                
                if step % 10 == 0:
                    current_time = time.time()
                    elapsed_time = current_time - last_log_time
                    tok_per_sec = (10 * tokens_per_step) / elapsed_time if elapsed_time > 0 else 0
                    
                    print(f"Step {step:05d} | Loss: {loss.item():.4f} | ⚡ 速度: {tok_per_sec:,.0f} tok/s | 📈 累计: {total_tokens_processed / 1e6:.2f} M")
                    
                    writer.add_scalar("Performance/Tokens_per_sec", tok_per_sec, step)
                    writer.add_scalar("Performance/Total_Tokens_M", total_tokens_processed / 1e6, step)
                    last_log_time = time.time()
                
                if step % 1000 == 0:
                    save_path = os.path.join(ckpt_dir, f"auto_ckpt_step_{step}.pt")
                    torch.save({"model_state": model.state_dict(), "step": step}, save_path)
                    print(f" [💾] 存档已写入: {save_path}")
                    
    except KeyboardInterrupt:
        print("\n🛑 收到中止信号 (Ctrl+C)，正在安全保存 final.pt...")
        torch.save({"model_state": model.state_dict(), "step": step}, os.path.join(ckpt_dir, "final.pt"))
        print("✅ 保存完毕！可以安全退出。")

if __name__ == "__main__":
    train()
