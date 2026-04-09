import os
import torch
import torch.nn.functional as F
import sys
import time

#  直接导入你原汁原味的模型架构！
from LNNModel import FluxPropLanguageModel

# ==========================================
# 硬件与架构超参数
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 1024     
RANK = 128            
NUM_LAYERS = 8        

# ==========================================
# 重建母语字典 (CharTokenizer)
# ==========================================
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]
    def decode(self, tokens):
        if isinstance(tokens, int):
            return self.itos.get(tokens, "?")
        return ''.join([self.itos.get(tok, "?") for tok in tokens])
    def vocab_size(self):
        return len(self.chars)

print(" 正在读取语料库以重建分词器字典...")
try:
    with open("train_en.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer = CharTokenizer(raw_text)
    VOCAB_SIZE = tokenizer.vocab_size()
    print(f" 字典重建成功: {VOCAB_SIZE}")
except FileNotFoundError:
    print(" 找不到 train_en.txt！请将其放在同一目录下。")
    sys.exit(1)

# ==========================================
# 🚀 采样器与生成引擎
# ==========================================
def sample_top_p(logits, top_p=0.9, temperature=0.7):
    # 如果 logits 是 3D 的 [batch, seq, vocab]，提取最后一步
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def generate(model, prompt_text, max_gen_len=500, temperature=0.7, top_p=0.9):
    model.eval()
    
    input_ids = tokenizer.encode(prompt_text)
    if not input_ids:
        print("\n 错误: 你的 Prompt 包含的字符都不在训练集字典里！")
        return
        
    h_states = None # 让模型自己初始化隐状态
    
    print(f"\n[Prompt]: {prompt_text}", end="", flush=True)
    
    with torch.no_grad():
        # 1. 预热 (Prefill)
        for i in range(len(input_ids) - 1):
            # 形状设为 [1, 1] 匹配你的模型输入 (batch=1, seq_len=1)
            x_t = torch.tensor([[input_ids[i]]], device=DEVICE)
            _, h_states = model(x_t, h_states)
            
        current_token = torch.tensor([[input_ids[-1]]], device=DEVICE)
        
        # 2. 生成 (Decode)
        for _ in range(max_gen_len):
            logits, h_states = model(current_token, h_states)
            next_token = sample_top_p(logits, top_p, temperature)
            
            gen_char = tokenizer.decode(next_token.item())
            print(gen_char, end="", flush=True)
            time.sleep(0.02)
            
            # 更新下一步的输入，保持 [1, 1] 形状
            current_token = next_token.view(1, 1)

# ==========================================
#主入口
# ==========================================
if __name__ == "__main__":
    print("⏳ 正在唤醒 FluxProp 动力学系统...")
    
    #  使用你原版的模型类进行初始化！
    model = FluxPropLanguageModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        num_layers=NUM_LAYERS, 
        device=DEVICE
    ).to(DEVICE)
    
    checkpoint_path = "checkpoints/full_final/absolute_final.pt" 
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        raw_state_dict = ckpt.get("model_state", ckpt)
        
        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            k = k.replace("_orig_mod.", "")
            k = k.replace("module.", "") 
            clean_state_dict[k] = v
            
        # 强行加载并获取诊断信息
        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        print("\n=== 对齐报告 ===")
        print(f"缺失项: {len(missing)} (如果全是 layer_4 到 7 的参数，这是正常的，因为它们未被训练)")
        print(f"未使用项: {len(unexpected)}")
        print("=========================\n")
        
    except FileNotFoundError:
        print(f" 找不到权重文件 {checkpoint_path}")
        sys.exit(1)
        
    print("-" * 50)
    prompt = "Deep in the forest, there was a "
    generate(model, prompt_text=prompt, max_gen_len=400, temperature=0.7)
    print("\n\n" + "=" * 50)
    print(" 生成结束。")