# FluxProp: Stateful-Liquid-Markovian-Language-Model

FluxProp is an experimental, highly ambitious language model architecture. It breaks the traditional dichotomy between Transformers and standard RNNs by deeply integrating continuous-time dynamics into a spatial residual framework.

---

## 🚀 Quick Start

**Prerequisites:** Ensure you have PyTorch 2.x and TensorBoard installed.

**Step 1: Download & Prepare Dataset**
```bash
python download_ultimate_wikitext.py
```

**Step 2: Layer-wise Curriculum Pre-training**
Train the architecture layer by layer to solidify the Spatio-Temporal Jacobian flow. Start from layer 0 up to layer 7:
```bash
python train_layer.py --layer 0
python train_layer.py --layer 1
# ... continue up to layer 7
python train_layer.py --layer 7
```

**Step 3: Global Full-Fusion Fine-Tuning**
Once all layers are solidified, unfreeze all parameters for global optimization at a low learning rate:
```bash
python train_full.py
```

**Step 4: Inference & Text Generation**
Load the final checkpoint to interactively generate text with continuous memory states (`h_states`):
```bash
python generate.py
```

---

### 1. Core Architecture & Theoretical Foundations

FluxProp is grounded in a rigorous joint spatio-temporal Jacobian analysis of continuous-time dynamics. The architecture operates with a complexity of $\mathcal{O}(d^2)$ for temporal updates and $\mathcal{O}(dr + r^2)$ for routing per layer.

#### 1.1 Temporal Liquid Update & Spatial Residuals (ReZero Optimized)
The evolution of the hidden state is governed by a nonlinear ODE. Discretized via the Forward Euler method, the temporal update for layer $l$ is strictly formulated as:

$$ \boldsymbol{h}_t^{(l)} = \left(\boldsymbol{1} - \sigma(\boldsymbol{\alpha}_{\text{raw}})\right) \odot \boldsymbol{h}_{t-1}^{(l)} + \tanh\left( \boldsymbol{W}^{(l)} \boldsymbol{h}_{t-1}^{(l)} + \boldsymbol{U}^{(l)} \boldsymbol{x}_t^{(l)} + \boldsymbol{b}^{(l)} \right) $$

In this implementation, $\boldsymbol{\alpha}_{\text{raw}}$ is a learnable vector parameter constrained by a Sigmoid function, allowing for channel-wise independent decay rates ($\odot$ denotes the Hadamard product). To guarantee a lossless gradient highway across spatial depth $L$, we incorporate a ReZero-inspired spatial residual correction:

$$ \boldsymbol{y}_t^{(l)} = \gamma \tilde{\boldsymbol{h}}_t^{(l)} + \text{Proj}(\boldsymbol{x}_t^{(l)}) $$

With the scalar $\gamma$ initialized to $0$, the spatial Jacobian at initialization rigorously equals the identity matrix $\boldsymbol{I}$, ensuring stable gradient flow during the early stages of training.

#### 1.2 Low-Rank Markovian State Routing (LSR)
Constructing a Markov transition matrix directly in $\mathbb{R}^d$ space imposes $\mathcal{O}(d^2)$ overhead. To resolve this, we project the state into a bottleneck space $r \ll d$:

$$ \boldsymbol{q}_t = \boldsymbol{W}_Q \boldsymbol{h}_t, \quad \boldsymbol{k}_t = \boldsymbol{W}_K \boldsymbol{h}_t, \quad \boldsymbol{v}_t = \boldsymbol{W}_V \boldsymbol{h}_t \quad \in \mathbb{R}^r $$

The Markovian transition matrix $\boldsymbol{M}_t$ is constructed using the scaled dot-product form, normalized across columns to form a valid Right Stochastic Matrix:

$$ \boldsymbol{M}_t = \text{Softmax}\left( \frac{\boldsymbol{q}_t \boldsymbol{k}_t^\top}{\sqrt{r}}, \dim=-1 \right) $$

The flux is routed through this bottleneck and projected back to $\mathbb{R}^d$:

$$ \tilde{\boldsymbol{h}}_t^{(l)} = \boldsymbol{W}_O (\boldsymbol{M}_t \boldsymbol{v}_t) $$

#### 1.3 Theorem 1: Emergence of Markovian Gradients
A key contribution of FluxProp is mathematically proving that routing in the forward pass intrinsically generates Markov-weighted gradients during Backpropagation Through Time (BPTT). 

The analytic gradient with respect to the pre-routed state $\boldsymbol{h}_t$ rigorously converges to:

$$ \nabla_{\boldsymbol{h}_t} \mathcal{L} = \underbrace{\boldsymbol{W}_V^\top \boldsymbol{M}_t^\top \boldsymbol{W}_O^\top (\nabla_{\tilde{\boldsymbol{h}}_t} \mathcal{L})}_{\text{Main Term: Markovian Transpose Weighting}} + \underbrace{\boldsymbol{J}_{\text{Softmax}}(\boldsymbol{h}_t)^\top (\nabla_{\tilde{\boldsymbol{h}}_t} \mathcal{L})}_{\text{Correction Term: Probabilistic Manifold Reshaping}} $$

The emergence of the $\boldsymbol{M}_t^\top$ term demonstrates an elegant system duality: the Markovian transition probabilities naturally re-weight the backward gradient flow inversely, mathematically rendering heuristic backward hooks obsolete.

### 2. Training Strategy: Layer-wise Staged Pre-training

To maintain stability across the non-linear Jacobian flow, FluxProp utilizes a staged training pipeline:
1. **Layer-wise Solidification**: Freeze all layers except the target layer (starting from Layer 0) to solidify basic feature extraction.
2. **Progressive Unfreezing**: Gradually introduce higher layers into the training loop, inheriting previous knowledge.
3. **Full Fusion**: Finally, unfreeze all parameters for global optimization at a low learning rate to achieve system-wide resonance.

---

# FluxProp: 有状态的液态马尔可夫语言模型

FluxProp 是一种极具野心的实验性语言模型架构。它通过将连续时间动力学深度整合进空间残差框架，打破了 Transformer 与 RNN 之间的界限。

---

## 🚀 快速开始运行指令

**环境准备：** 请确保安装了 PyTorch 2.x 以及 TensorBoard。

**第一步：下载并提取核心训练集**
```bash
python download_ultimate_wikitext.py
```
*(注：如果需要进行微调，请自行在目录下准备 `sft.txt` 语料)*

**第二步：逐层固化预训练 (Layer-wise Curriculum)**
为了稳定深层的时空 Jacobian 流，请从第 0 层开始，逐步完成所有层的物理固化训练（训练过程中会自动继承下层权重）：
```bash
python train_layer.py --layer 0
python train_layer.py --layer 1
# ... 依次训练至第 7 层
python train_layer.py --layer 7
```

**第三步：全量联合微调 (Global Fine-tuning)**
底层知识固化完成后，启动全参解冻的全局联合优化，冲击系统动态共振：
```bash
python train_full.py
```

**第四步：运行推理生成 (Inference)**
加载最终的绝对权重，启动带有深层状态记忆（`h_states`）的时空文本连续推演生成：
```bash
python generate.py
```

---

### 1. 核心架构与理论基础

FluxProp 基于严谨的时空联合 Jacobian 分析。每一层的时间更新复杂度为 $\mathcal{O}(d^2)$，路由复杂度为 $\mathcal{O}(dr + r^2)$。

#### 1.1 时间液态更新与空间残差 (ReZero 优化)
隐藏状态的演化受非线性 ODE 支配。通过前向欧拉法离散化后，第 $l$ 层的差分更新公式严谨表述为：

$$ \boldsymbol{h}_t^{(l)} = \left(\boldsymbol{1} - \sigma(\boldsymbol{\alpha}_{\text{raw}})\right) \odot \boldsymbol{h}_{t-1}^{(l)} + \tanh\left( \boldsymbol{W}^{(l)} \boldsymbol{h}_{t-1}^{(l)} + \boldsymbol{U}^{(l)} \boldsymbol{x}_t^{(l)} + \boldsymbol{b}^{(l)} \right) $$

在这里，$\boldsymbol{\alpha}_{\text{raw}}$ 是经 Sigmoid 函数约束的可学习向量，支持特征通道级的独立衰减（$\odot$ 表示哈达玛积）。为了保证空间深度 $L$ 上的无损梯度传输，我们引入了 ReZero 空间残差修正：

$$ \boldsymbol{y}_t^{(l)} = \gamma \tilde{\boldsymbol{h}}_t^{(l)} + \text{Proj}(\boldsymbol{x}_t^{(l)}) $$

通过将标量 $\gamma$ 初始化为 $0$，初始状态下的跨层空间 Jacobian 连乘矩阵严格等价于单位阵 $\boldsymbol{I}$，从物理底层确保了训练初期深层梯度的有效传递。

#### 1.2 低秩马尔可夫状态路由 (LSR)
为了避免 $\mathcal{O}(d^2)$ 的极端高维计算开销，我们将状态投影至低秩瓶颈空间 $r \ll d$：

$$ \boldsymbol{q}_t = \boldsymbol{W}_Q \boldsymbol{h}_t, \quad \boldsymbol{k}_t = \boldsymbol{W}_K \boldsymbol{h}_t, \quad \boldsymbol{v}_t = \boldsymbol{W}_V \boldsymbol{h}_t \quad \in \mathbb{R}^r $$

利用缩放点积形式构造马尔可夫转移矩阵 $\boldsymbol{M}_t$，并在列方向进行严格的 Softmax 归一化，使其满足右随机矩阵性质：

$$ \boldsymbol{M}_t = \text{Softmax}\left( \frac{\boldsymbol{q}_t \boldsymbol{k}_t^\top}{\sqrt{r}}, \dim=-1 \right) $$

随后通过输出投影将路由分配后的通量映射回高维主空间：

$$ \tilde{\boldsymbol{h}}_t^{(l)} = \boldsymbol{W}_O (\boldsymbol{M}_t \boldsymbol{v}_t) $$

#### 1.3 定理 1：马尔可夫梯度的自发涌现
FluxProp 的核心理论采用严密的张量微积分证明了：前向传播中的概率路由机制，会在 BPTT（随时间反向传播）过程中通过自动微分框架自发涌现出马尔可夫加权梯度流。

预路由隐状态 $\boldsymbol{h}_t$ 的完整解析反向梯度严格收敛于：

$$ \nabla_{\boldsymbol{h}_t} \mathcal{L} = \underbrace{\boldsymbol{W}_V^\top \boldsymbol{M}_t^\top \boldsymbol{W}_O^\top (\nabla_{\tilde{\boldsymbol{h}}_t} \mathcal{L})}_{\text{主导项: 马尔可夫概率转置加权}} + \underbrace{\boldsymbol{J}_{\text{Softmax}}(\boldsymbol{h}_t)^\top (\nabla_{\tilde{\boldsymbol{h}}_t} \mathcal{L})}_{\text{调节项: 概率流形二阶重塑}} $$

主导项中 $\boldsymbol{M}_t^\top$ 的涌现展现了绝佳的系统动力学对偶性：反向传播的误差信号会严格按照前向转移概率的**转置矩阵（即逆向概率拓扑）**自动倒灌，这在数学上完全消除了对传统 RNN 中启发式反向钩子（Backward Hooks）的依赖。

### 2. 训练策略：逐层课程预训练

由于深层 Jacobian 流的极度复杂性，项目采用了“物理锁死”的逐层固化训练策略：
1. **单层固化**：从 Layer 0 开始，物理锁死（冻结）其他层，专注固化当前层的基础特征提取能力，防止底层特征污染。
2. **渐进解冻**：逐层向上解冻并注入训练数据，利用残差自动继承底层成果。
3. **全量融合**：最后以极低学习率全局解冻物理参数，冲击全局联合 Jacobian 流的动态共振优化。
```
