import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FluxPropLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, rank=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # ==========================================
        # 1. 液态动力学 (Temporal Liquid Update)
        # ==========================================
        self.U = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 🛠️ 修复：将 alpha 扩展为 hidden_dim 维度，允许不同特征拥有独立遗忘率
        self.alpha_raw = nn.Parameter(torch.zeros(hidden_dim))

        # ==========================================
        # 2. 低秩马尔可夫路由 (Low-Rank Markovian Routing)
        # ==========================================
        self.W_Q = nn.Linear(hidden_dim, rank, bias=False)
        self.W_K = nn.Linear(hidden_dim, rank, bias=False)
        self.W_V = nn.Linear(hidden_dim, rank, bias=False)
        self.W_O = nn.Linear(rank, hidden_dim, bias=False)

        # ==========================================
        # 3. 空间残差保底 (Spatial Residual with ReZero)
        # ==========================================
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

    def forward(self, h_prev, x_t):
        # 1. 液态时间更新 
        alpha = torch.sigmoid(self.alpha_raw) # [hidden_dim]
        liquid_stimulus = torch.tanh(self.W(h_prev) + self.U(x_t))
        # 广播机制自动对齐 [B, hidden_dim]
        h_t = (1.0 - alpha) * h_prev + liquid_stimulus

        # 2. 低秩空间投影
        q_t = self.W_Q(h_t)  # [B, R]
        k_t = self.W_K(h_t)  # [B, R]
        v_t = self.W_V(h_t)  # [B, R]

        q_t_exp = q_t.unsqueeze(-1)
        k_t_exp = k_t.unsqueeze(1)

        # 3. 马尔可夫概率转移矩阵
        M_raw = torch.bmm(q_t_exp, k_t_exp) / math.sqrt(self.rank)
        M_t = F.softmax(M_raw, dim=-1)

        # 4. 低秩通量路由与升维
        v_t_exp = v_t.unsqueeze(-1)
        routed_v = torch.bmm(M_t, v_t_exp).squeeze(-1) 
        h_tilde = self.W_O(routed_v)

        # 5. 空间残差输出
        y_t = self.gamma * h_tilde + self.res_proj(x_t)

        return h_t, y_t

class FluxPropLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, rank=64, num_layers=8, device="cpu"):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
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
        
        # 权重绑定
        self.lm_head.weight = self.embedding.weight

        # 压缩初始方差，强制在 5.0 附近起步
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if hasattr(self.lm_head, 'bias') and self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    # 🛠️ 修复：支持传入外部 h_states，实现序列间的记忆连续性
    def forward(self, input_ids, h_states=None):
        x = self.embedding(input_ids.to(self.device))
        batch_size, seq_len, _ = x.shape
        
        if h_states is None:
            h_states = [torch.zeros(batch_size, self.hidden_dim, device=self.device) for _ in range(self.num_layers)]
            
        outputs = []

        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for i, layer in enumerate(self.layers):
                h_t, y_t = layer(h_states[i], layer_input)
                h_states[i] = h_t
                layer_input = y_t
                
            outputs.append(layer_input.unsqueeze(1))
            
        sequence_output = torch.cat(outputs, dim=1)
        logits = self.lm_head(sequence_output)
        
        return logits, h_states