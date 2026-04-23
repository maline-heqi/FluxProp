import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def verify_markovian_bptt():
    # 为了热力图可视化更清晰，我们使用 32 维的 Rank
    B, r = 1, 32
    torch.manual_seed(42)
    
    # ==========================================
    # 1. 模拟前向传播中的状态投影
    # ==========================================
    # q, k 决定了马尔可夫转移概率
    q = torch.randn(B, r, 1)
    k = torch.randn(B, 1, r)
    
    # 生成前向马尔可夫转移矩阵 M_t (脱离计算图，因为我们专注验证对 v 的偏导)
    M_raw = torch.bmm(q, k) / (r ** 0.5)
    M_t = F.softmax(M_raw, dim=-1).detach() # [1, r, r]
    M_t_matrix = M_t.squeeze(0)             # [r, r]
    
    # ==========================================
    # 2. 核心：通过 BPTT 逆向提取“经验反向路由矩阵”
    # ==========================================
    empirical_backward_routing = torch.zeros(r, r)
    
    # 利用单位矩阵的列向量作为独立的上游梯度，逐个注入
    I = torch.eye(r)
    
    for i in range(r):
        v_t = torch.randn(B, r, 1, requires_grad=True)
        
        # 前向通量路由: routed_v = M_t @ v_t
        routed_v = torch.bmm(M_t, v_t)
        
        # 提取第 i 个特征维度的上游梯度
        upstream_grad = I[:, i].unsqueeze(0).unsqueeze(-1) # [1, r, 1]
        
        # 触发 BPTT
        routed_v.backward(upstream_grad)
        
        # ✅ 修复核心：PyTorch 返回的是 M^T 的第 i 列，所以必须填入第 i 列！
        empirical_backward_routing[:, i] = v_t.grad.squeeze()
        
        # 清空梯度，保持严谨，准备下一次探测
        v_t.grad = None

    # ==========================================
    # 3. 理论验证对比
    # ==========================================
    theoretical_backward_routing = M_t_matrix.T
    
    # 计算绝对误差
    max_error = torch.max(torch.abs(empirical_backward_routing - theoretical_backward_routing)).item()
    print(f"🚀 定理 1 验证报告 🚀")
    print(f"最大绝对误差 (Empirical vs Theoretical M_t^T): {max_error:.8e}")
    
    if max_error < 1e-6:
        print("✅ 验证成功！BPTT 的自动求导天然重构了马尔可夫转移矩阵的转置。")
        
    # ==========================================
    # 4. 生成论文级可视化热力图
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图 1: 前向马尔可夫矩阵 M_t
    im1 = axes[0].imshow(M_t_matrix.numpy(), cmap='viridis', aspect='auto')
    axes[0].set_title(r"Forward Flux Routing ($M_t$)", fontsize=14)
    axes[0].set_xlabel("Target State ($v_t$ dimension)")
    axes[0].set_ylabel("Source State ($\tilde{h}_t$ dimension)")
    fig.colorbar(im1, ax=axes[0])
    
    # 图 2: BPTT 测得的反向梯度路由流
    im2 = axes[1].imshow(empirical_backward_routing.numpy(), cmap='viridis', aspect='auto')
    axes[1].set_title("Empirical Backward Gradient Flow", fontsize=14)
    axes[1].set_xlabel("Gradient from ($\tilde{h}_t$)")
    axes[1].set_ylabel("Gradient to ($v_t$)")
    fig.colorbar(im2, ax=axes[1])
    
    # 图 3: 理论转置 M_t^T
    im3 = axes[2].imshow(theoretical_backward_routing.numpy(), cmap='viridis', aspect='auto')
    axes[2].set_title(r"Theoretical Transpose ($M_t^T$)", fontsize=14)
    axes[2].set_xlabel("Target State ($\tilde{h}_t$ dimension)")
    axes[2].set_ylabel("Source State ($v_t$ dimension)")
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('theorem1_markov_emergence.png', dpi=300)
    print("📸 热力图对比已保存为 theorem1_markov_emergence.png")

if __name__ == "__main__":
    verify_markovian_bptt()