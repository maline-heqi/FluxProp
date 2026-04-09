# FluxProp: Stateful-Liquid-Markovian-Language-Model

FluxProp is an experimental, highly ambitious language model architecture. It breaks the traditional dichotomy between Transformers and standard RNNs by deeply integrating continuous-time dynamics into a spatial residual framework.

### 1. Core Architecture & Theoretical Foundations

FluxProp is grounded in a rigorous joint spatio-temporal Jacobian analysis of continuous-time dynamics. The architecture operates with a complexity of O(d^2) for temporal updates and O(dr + r^2) for routing per layer.

#### 1.1 Temporal Liquid Update & Spatial Residuals (ReZero Optimized)
The evolution of the hidden state is governed by a nonlinear ODE. Discretized via the Forward Euler method, the temporal update for layer l is:

$$h_t = (1 - sigmoid(alpha)) * h_{t-1} + tanh(W * h_{t-1} + U * x_t + b)$$

In this implementation, alpha is a learnable vector parameter, allowing for channel-wise independent decay rates. To guarantee a lossless gradient highway across spatial depth L, we incorporate a ReZero-inspired spatial residual correction:

y_t = gamma * h_tilde_t + Proj(x_t)

With gamma initialized to 0, the spatial Jacobian at initialization remains an identity matrix I, ensuring stable gradient flow during the early stages of training.

#### 1.2 Low-Rank Markovian State Routing (LSR)
Constructing a Markov transition matrix directly in R^d space imposes O(d^2) overhead. To resolve this, we project the state into a bottleneck space r << d:

q_t = W_Q * h_t,  k_t = W_K * h_t,  v_t = W_V * h_t

The Markovian transition matrix M_t is constructed using the scaled dot-product form:

M_t = Softmax( (q_t * k_t^T) / sqrt(r) )

The flux is routed through this bottleneck and projected back to R^d:

h_tilde_t = W_O * (M_t * v_t)

#### 1.3 Theorem 1: Emergence of Markovian Gradients
A key contribution of FluxProp is proving that routing in the forward pass intrinsically generates Markov-weighted gradients during Backpropagation Through Time (BPTT). 

The gradient with respect to the pre-routed state rigorously converges to:

dL/dh_t = M_t^T * (dL/d_h_tilde_t) + Jacobian_Softmax(h_t)^T * (dL/d_h_tilde_t)

The dominance of the M_t^T term demonstrates that the Markovian transition probabilities naturally re-weight the gradient flow, rendering heuristic backward hooks obsolete.

### 2. Training Strategy: Layer-wise Staged Pre-training

To maintain stability across the non-linear Jacobian flow, FluxProp utilizes a staged training pipeline:
1. Layer-wise Solidification: Freeze all layers except the target layer (starting from Layer 0) to solidify basic feature extraction.
2. Progressive Unfreezing: Gradually introduce higher layers into the training loop.
3. Full Fusion: Finally, unfreeze all parameters for global optimization at a low learning rate to achieve system-wide resonance.

---

# FluxProp: 有状态液态马尔可夫语言模型

FluxProp 是一种极具野心的实验性语言模型架构。它通过将连续时间动力学深度整合进空间残差框架，打破了 Transformer 与 RNN 之间的界限。

### 1. 核心架构与理论基础

FluxProp 基于严谨的时空联合 Jacobian 分析。每一层的时间更新复杂度为 O(d^2)，路由复杂度为 O(dr + r^2)。

#### 1.1 时间液态更新与空间残差 (ReZero 优化)
隐藏状态的演化受非线性 ODE 支配。通过前向欧拉法离散化后，l 层的更新公式为：

h_t = (1 - sigmoid(alpha)) * h_{t-1} + tanh(W * h_{t-1} + U * x_t + b)

在这里，alpha 是可学习的向量参数，支持通道级的独立衰减。为了保证空间深度 L 上的无损梯度传输，我们引入了 ReZero 空间残差修正：

y_t = gamma * h_tilde_t + Proj(x_t)

通过将 gamma 初始化为 0，初始状态下的空间 Jacobian 矩阵为单位矩阵 I，确保了训练初期的稳定性。

#### 1.2 低秩马尔可夫状态路由 (LSR)
为了避免 O(d^2) 的计算开销，我们将状态投影至低秩瓶颈空间 r << d：

q_t = W_Q * h_t,  k_t = W_K * h_t,  v_t = W_V * h_t

利用缩放点积形式构造马尔可夫转移矩阵 M_t：

M_t = Softmax( (q_t * k_t^T) / sqrt(r) )

随后通过输出投影将通量映射回高维空间：

h_tilde_t = W_O * (M_t * v_t)

#### 1.3 定理 1：马尔可夫梯度的自发涌现
FluxProp 的核心理论证明了前向传播中的路由机制会在 BPTT 过程中自动产生马尔可夫加权梯度。

预路由状态的梯度收敛于：

dL/dh_t = M_t^T * (dL/d_h_tilde_t) + Jacobian_Softmax(h_t)^T * (dL/d_h_tilde_t)

M_t^T 项的出现意味着梯度流会根据前向转移概率自动分配权重，这使得启发式的反向钩子（Backward Hooks）不再必要。

### 2. 训练策略：逐层课程预训练

由于深层 Jacobian 流的复杂性，项目采用了逐层固化训练策略：
1. 单层固化：从 Layer 0 开始，冻结其他层以稳定基础特征空间。
2. 渐进解冻：逐层向上解冻并注入训练。
3. 全量融合：最后以极低学习率解冻全局参数，实现整个 8 层动力学系统的全局优化。
\frac{\partial\mathcal{L}}{\partial h_{t}} = M_{t}^{\top}(\nabla_{\tilde{h}_{t}}\mathcal{L}) + Jacobian_{Softmax}(h_{t})^{\top}(\nabla_{\tilde{h}_{t}}\mathcal{L})
$$

The dominant term $M_{t}^{\top}\nabla_{\tilde{h}_{t}}\mathcal{L}$ demonstrates that the Markovian transition probability naturally re-weights the gradient flow, mathematically rendering heuristic backward hooks obsolete.
