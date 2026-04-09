import os
import time
import math
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

# 假设你的模型类定义在 LNNModel.py 中
from LNNModel import FluxPropLanguageModel

# ==========================================
# 🚀 全局联调超参数 (The Master Configuration)
# ==========================================
SEQ_LEN = 256
HIDDEN_DIM = 1024
RANK = 128
# 🚨 注意：全量解冻显存占用极高。如果 24G 显存 OOM，请将 BATCH_SIZE 降至 128 或 192
BATCH_SIZE = 256  
# 🔴 关键：学习率必须极低，用于“文火慢炖”已有的层级特征
BASE_LR = 1e-5  
WARMUP_STEPS = 200
TOTAL_STEPS = 10000 
WEIGHT_DECAY = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

# ==========================================
# 📖 数据处理 (保持与 train_layer 一致)
# ==========================================
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    def encode(self, text): return [self.stoi[ch] for ch in text if ch in self.stoi]
    def decode(self, tokens): return ''.join([self.itos[tok] for tok in tokens])
    def vocab_size(self): return len(self.chars)

class StatefulDataLoader:
    def __init__(self, data_tensor, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        n_seqs = len(data_tensor) // batch_size
        self.data = data_tensor[:n_seqs * batch_size].view(batch_size, n_seqs)
        self.n_batches = (n_seqs - 1) // seq_len
    def __iter__(self):
        for i in range(self.n_batches):
            yield self.data[:, i*self.seq_len : (i+1)*self.seq_len], \
                  self.data[:, i*self.seq_len+1 : (i+1)*self.seq_len+1]

# ==========================================
# 🧠 训练核心逻辑
# ==========================================
def get_lr(step):
    # 带有 Warmup 的余弦退火学习率
    if step < WARMUP_STEPS:
        return BASE_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))

def train_full():
    # 1. 准备数据
    # ✅ 正确做法：用原维基百科语料固定词表架构，再编码微调数据
    print("📚 正在加载母语字典...")
    with open("train_en.txt", "r", encoding="utf-8") as f:
        base_text = f.read()
    tokenizer = CharTokenizer(base_text)  # 完美恢复 4979 的词汇维度和原始索引！
    
    print("📖 正在读取指令微调数据...")
    with open("sft.txt", "r", encoding="utf-8") as f:
        sft_text = f.read()
        
    # 编码 sft 数据 (CharTokenizer里的 if ch in self.stoi 会自动过滤掉极端罕见的生僻符号)
    encoded_sft = tokenizer.encode(sft_text)
    data = torch.tensor(encoded_sft, dtype=torch.long)
    loader = StatefulDataLoader(data, BATCH_SIZE, SEQ_LEN)
    
    print(f"📊 SFT Token总数: {len(data)}, 词表大小: {tokenizer.vocab_size()}")
    loader = StatefulDataLoader(data, BATCH_SIZE, SEQ_LEN)

    # 2. 初始化全量模型 (8层)
    model = FluxPropLanguageModel(
        vocab_size=tokenizer.vocab_size(),
        embed_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        num_layers=8,
        device=DEVICE
    ).to(DEVICE)

    # 3. 加载 Layer 07 的最终成果
    ckpt_path = "checkpoints/full_final/absolute_final.pt"
    if os.path.exists(ckpt_path):
        print(f"📦 正在加载 Layer 07 基础权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        # 兼容性处理：移除可能存在的分布式/编译前缀
        sd = ckpt.get("model_state", ckpt)
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        raise FileNotFoundError("必须先完成 Layer 07 的训练才能开启全量调优！")

    # 🔥 核心：解冻全部参数
    for param in model.parameters():
        param.requires_grad = True
    print("🔓 全局解冻完成。进入联合 Jacobian 流优化阶段。")

    # 4. 优化器配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()
    writer = SummaryWriter("runs/FluxProp_Full_V1")

    model.train()
    step = 0
    h_states = None
    start_time = time.time()

    print(f"🚀 ===================================================")
    print(f"🚀 终章开启：全量全局微调模式")
    print(f"🚀 初始 Loss 锚点: ~1.70 | 目标: 冲击 1.5x")
    print(f"🚀 ===================================================")

    try:
        while step < TOTAL_STEPS:
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                # 调整学习率
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # 维持隐状态连续性 (Detach防止计算图无限向后延伸)
                if h_states is not None:
                    h_states = [h.detach() for h in h_states]

                optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type="cuda"):
                    logits, h_states = model(x, h_states)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                scaler.scale(loss).backward()
                
                # 更加严格的梯度裁剪，防止全量解冻后的数值抖动
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                scaler.step(optimizer)
                scaler.update()

                step += 1
                if step % 10 == 0:
                    dt = time.time() - start_time
                    tok_per_sec = (10 * BATCH_SIZE * SEQ_LEN) / dt
                    print(f"Step {step:05d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | ⚡ {tok_per_sec:,.0f} tok/s")
                    writer.add_scalar("Full/Loss", loss.item(), step)
                    writer.add_scalar("Full/LR", lr, step)
                    start_time = time.time()

                if step % 1000 == 0:
                    save_path = f"checkpoints/ture_full_final/step_{step}.pt"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    print(f"💾 已保存周期性权重: {save_path}")

                if step >= TOTAL_STEPS: break

    except KeyboardInterrupt:
        print("\n🛑 收到中断信号，正在安全撤离并保存权重...")
    
    final_save = "checkpoints/Ture_full_final/ture_absolute_final.pt"
    os.makedirs(os.path.dirname(final_save), exist_ok=True)
    torch.save(model.state_dict(), final_save)
    print(f"✅ 恭喜！全量模型已就绪: {final_save}")

if __name__ == "__main__":
    train_full()