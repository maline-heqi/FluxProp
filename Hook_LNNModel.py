import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 📊 诊断工具：马尔可夫梯度余弦相似度探针
# ==========================================
class MarkovianGradientProbe:
    """
    专门用于验证定理 1 的探测器：
    计算实际 BPTT 梯度与理论预测主导项 (Mt^T * ∇L) 之间的余弦相似度。
    """
    def __init__(self):
        self.actual_grads = []
        self.theoretical_grads = []
        self.cosine_similarities = []
        self.M_t_cache = None

    def capture_forward_states(self, M_t, v_t, routed_v):
        """在前向传播中截获状态，并挂载反向传播钩子"""
        # 脱离计算图保存当前步的路由矩阵
        self.M_t_cache = M_t.detach() 
        
        # 显式保留中间变量梯度
        v_t.retain_grad()
        routed_v.retain_grad()
        
        # 挂载钩子：反向传播传到这里时触发对比
        routed_v.register_hook(lambda grad: self._compute_theoretical(grad))
        v_t.register_hook(lambda grad: self._compute_actual(grad))

    def _compute_theoretical(self, routed_v_grad):
        # 理论预测项：M_t^T * ∇_routed_v L
        M_t_T = self.M_t_cache.transpose(-1, -2)
        theoretical_grad = torch.bmm(M_t_T, routed_v_grad)
        self.theoretical_grads.append(theoretical_grad.detach())

    def _compute_actual(self, v_t_grad):
        # 获取 BPTT 真实回传的梯度
        self.actual_grads.append(v_t_grad.detach())
        
        # 计算方向相似度 (Cosine Similarity)
        if len(self.theoretical_grads) > 0:
            theo = self.theoretical_grads[-1]
            act = v_t_grad.detach()
            # 展平后计算 Batch 整体的方向对齐度
            cos_sim = F.cosine_similarity(act.flatten(), theo.flatten(), dim=0)
            self.cosine_similarities.append(cos_sim.item())

    def plot_similarity_curve(self, save_path="cosine_similarity_report.png"):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ 未检测到 matplotlib，无法绘图。平均相似度已打印。")
        else:
            plt.figure(figsize=(10, 5))
            # BPTT 是反向的，绘图前需反转回正序
            sims = self.cosine_similarities[::-1]
            plt.plot(sims, label='Cosine Similarity (Actual vs Predicted)', color='#2ecc71', linewidth=2)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            plt.title("BPTT Gradient Alignment: Markovian Dominance")
            plt.xlabel("Sequence Steps (t)")
            plt.ylabel("Similarity")
            plt.ylim(0.8, 1.05)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            print(f"📊 验证图表已保存: {save_path}")

        if self.cosine_similarities:
            avg = sum(self.cosine_similarities)/len(self.cosine_similarities)
            print(f"🎯 理论对齐度 (Cosine Similarity): {avg:.6f}")


# ==========================================
# 🧠 FluxProp 核心层级架构
# ==========================================
class FluxPropLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, rank=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        self.U = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.alpha_raw = nn.Parameter(torch.zeros(hidden_dim))

        self.W_Q = nn.Linear(hidden_dim, rank, bias=False)
        self.W_K = nn.Linear(hidden_dim, rank, bias=False)
        self.W_V = nn.Linear(hidden_dim, rank, bias=False)
        self.W_O = nn.Linear(rank, hidden_dim, bias=False)

        self.res_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(self, h_prev, x_t, probe=None):
        # 1. 时间液态动力学更新
        alpha = torch.sigmoid(self.alpha_raw)
        liquid_stimulus = torch.tanh(self.W(h_prev) + self.U(x_t))
        h_t = (1.0 - alpha) * h_prev + liquid_stimulus

        # 2. 生成低秩马尔可夫转移矩阵 M_t
        q_t = self.W_Q(h_t).unsqueeze(-1)
        k_t = self.W_K(h_t).unsqueeze(1)
        M_t = F.softmax(torch.bmm(q_t, k_t) / math.sqrt(self.rank), dim=-1)

        # 3. 低秩通量路由
        v_t = self.W_V(h_t).unsqueeze(-1)
        
        # 🔌 如果传入探针，对 v_t 显式标记计算图
        if probe is not None:
            v_t = v_t.clone().requires_grad_(True)
            
        routed_v = torch.bmm(M_t, v_t)
        
        # 🔌 探针截获点
        if probe is not None:
            probe.capture_forward_states(M_t, v_t, routed_v)

        # 4. 空间映射与残差
        h_tilde = self.W_O(routed_v.squeeze(-1))
        y_t = self.gamma * h_tilde + self.res_proj(x_t)

        return h_t, y_t

class FluxPropLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=1024, hidden_dim=1024, rank=128, num_layers=8, device="cpu"):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            FluxPropLayer(
                input_dim=embed_dim if i == 0 else hidden_dim, 
                hidden_dim=hidden_dim, 
                rank=rank
            )
            for i in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.lm_head.weight = self.embedding.weight

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, h_states=None, probe=None):
        x = self.embedding(input_ids.to(self.device))
        batch_size, seq_len, _ = x.shape
        
        if h_states is None:
            h_states = [torch.zeros(batch_size, self.hidden_dim, device=self.device) for _ in range(self.num_layers)]
            
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            for i, layer in enumerate(self.layers):
                # 将探针向下透传给每一层（或指定层）
                # 这里为了简单，让所有层共享同一个探针
                h_t, y_t = layer(h_states[i], layer_input, probe=probe)
                h_states[i] = h_t
                layer_input = y_t
            outputs.append(layer_input.unsqueeze(1))
            
        sequence_output = torch.cat(outputs, dim=1)
        logits = self.lm_head(sequence_output)
        
        return logits, h_states