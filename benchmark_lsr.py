import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# 激活 Tensor Core 加速
torch.set_float32_matmul_precision('high')

# ==========================================
# 1. 对照组：全矩阵马尔可夫路由 O(d^2)
# ==========================================
class FullMatrix_Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        # 在全维空间构造转移矩阵，等价于 r = d
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)
        self.d = d

    def forward(self, h):
        q = self.W_Q(h).unsqueeze(-1) # [B, d, 1]
        k = self.W_K(h).unsqueeze(1)  # [B, 1, d]
        v = self.W_V(h).unsqueeze(-1) # [B, d, 1]
        
        # 显存杀手：[B, d, d] 的全维马尔可夫转移矩阵
        M = torch.softmax(torch.bmm(q, k) / math.sqrt(self.d), dim=-1) 
        return self.W_O(torch.bmm(M, v).squeeze(-1))

# ==========================================
# 2. 实验组：FluxProp 低秩马尔可夫路由 (LSR) O(dr + r^2)
# ==========================================
class LSR_Layer(nn.Module):
    def __init__(self, d, r=64):
        super().__init__()
        self.W_Q = nn.Linear(d, r, bias=False)
        self.W_K = nn.Linear(d, r, bias=False)
        self.W_V = nn.Linear(d, r, bias=False)
        self.W_O = nn.Linear(r, d, bias=False)
        self.r = r

    def forward(self, h):
        q = self.W_Q(h).unsqueeze(-1) # [B, r, 1]
        k = self.W_K(h).unsqueeze(1)  # [B, 1, r]
        v = self.W_V(h).unsqueeze(-1) # [B, r, 1]
        
        # 高效路由：[B, r, r] 的低秩马尔可夫转移矩阵
        M = torch.softmax(torch.bmm(q, k) / math.sqrt(self.r), dim=-1)
        return self.W_O(torch.bmm(M, v).squeeze(-1))

# ==========================================
# 3. 核心压测引擎
# ==========================================
def run_benchmark(model_class, d, r=64, batch_size=128, seq_len=64, iters=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("⚠️ 警告：未检测到 GPU，正在使用 CPU 运行测试！性能数据可能不准确。")
        
    model = model_class(d) if model_class == FullMatrix_Layer else model_class(d, r)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 模拟外部输入 (连续时间流刺激)
    x = torch.randn(seq_len, batch_size, d, device=device)
    
    try:
        # --- Warmup 预热 ---
        for _ in range(5):
            h = torch.zeros(batch_size, d, device=device)
            for t in range(seq_len):
                h = model(h) + x[t]
            loss = h.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
        # --- 正式测速 ---
        start_time = time.time()
        for _ in range(iters):
            h = torch.zeros(batch_size, d, device=device)
            # 模拟 BPTT 沿时间展开
            for t in range(seq_len):
                h = model(h) + x[t]
            loss = h.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        # 计算指标
        total_time = end_time - start_time
        total_tokens = batch_size * seq_len * iters
        tps = total_tokens / total_time
        max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        
        return tps, max_mem
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ 显存爆炸 (OOM) @ d={d}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return 0, 0
        else:
            raise e

# ==========================================
# 4. 执行测试与可视化绘图
# ==========================================
if __name__ == "__main__":
    dims = [1024, 2048, 4096]
    rank = 64
    
    results_lsr = {'tps': [], 'mem': []}
    results_full = {'tps': [], 'mem': []}
    
    print("🚀 开始 FluxProp 硬件级基准测试 (BPTT Continuous Routing)...")
    for d in dims:
        print(f"\n🧪 测试维度: d = {d} | Rank = {rank}")
        
        print("  ▶ 正在测试 [LSR 机制]...")
        lsr_tps, lsr_mem = run_benchmark(LSR_Layer, d, r=rank)
        results_lsr['tps'].append(lsr_tps)
        results_lsr['mem'].append(lsr_mem)
        print(f"    TPS: {lsr_tps:,.0f} tok/s | VRAM: {lsr_mem:.0f} MB")
        
        print("  ▶ 正在测试 [全矩阵机制]...")
        full_tps, full_mem = run_benchmark(FullMatrix_Layer, d)
        results_full['tps'].append(full_tps)
        results_full['mem'].append(full_mem)
        print(f"    TPS: {full_tps:,.0f} tok/s | VRAM: {full_mem:.0f} MB")

    # ================= 绘图 =================
    x = np.arange(len(dims))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 图 1：吞吐量 (越高越好)
    ax1.bar(x - width/2, results_full['tps'], width, label='Full-Matrix O(d²)', color='#e74c3c')
    ax1.bar(x + width/2, results_lsr['tps'], width, label=f'FluxProp LSR (r={rank})', color='#3498db')
    ax1.set_ylabel('Throughput (Tokens / Sec)')
    ax1.set_title('BPTT Throughput Comparison (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"d={d}" for d in dims])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 图 2：显存占用 (越低越好)
    ax2.bar(x - width/2, [m if m > 0 else 0 for m in results_full['mem']], width, label='Full-Matrix O(d²)', color='#e74c3c')
    ax2.bar(x + width/2, results_lsr['mem'], width, label=f'FluxProp LSR (r={rank})', color='#3498db')
    ax2.set_ylabel('Peak VRAM Usage (MB)')
    ax2.set_title('Memory Wall Comparison (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"d={d}" for d in dims])
    
    # 标记 OOM
    for i, mem in enumerate(results_full['mem']):
        if mem == 0:
            ax2.text(x[i] - width/2, 100, "OOM", color='white', ha='center', fontweight='bold')

    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('hardware_benchmark.png', dpi=300)
    print("\n✅ 测试完成！基准测试图表已保存为 hardware_benchmark.png")