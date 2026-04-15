````markdown
# FluxProp: Stateful-Liquid-Markovian-Language-Model

FluxProp is an experimental, highly ambitious language model architecture. It breaks the traditional dichotomy between Transformers and standard RNNs by deeply integrating continuous-time dynamics into a spatial residual framework.

*[阅读中文版说明 (Read this in Chinese)](#fluxprop-有状态的液态马尔可夫语言模型)*

---

## 🚀 Quick Start

**Prerequisites:** Ensure you have PyTorch 2.x and TensorBoard installed.

**Step 1: Download & Prepare Dataset**
```bash
python download_ultimate_wikitext.py
````

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

-----

## 1\. Core Architecture & Theoretical Foundations

FluxProp is grounded in a rigorous joint spatio-temporal Jacobian analysis of continuous-time dynamics. The architecture operates with a complexity of $\mathcal{O}(d^2)$ for temporal updates and $\mathcal{O}(dr + r^2)$ for routing per layer.

### 1.1 Temporal Liquid Update & Spatial Residuals (ReZero Optimized)

The evolution of the hidden state is governed by a nonlinear ODE. Discretized via the Forward Euler method, the temporal update for layer $l$ is strictly formulated as:

$$ \boldsymbol{h}_t^{(l)} = \left(\boldsymbol{1} - \sigma(\boldsymbol{\alpha}_{\text{raw}})\right) \odot \boldsymbol{h}_{t-1}^{(l)} + \tanh\left( \boldsymbol{W}^{(l)} \boldsymbol{h}_{t-1}^{(l)} + \boldsymbol{U}^{(l)} \boldsymbol{x}_t^{(l)} + \boldsymbol{b}^{(l)} \right) $$

In this implementation, $\boldsymbol{\alpha}_{\text{raw}}$ is a learnable vector parameter constrained by a Sigmoid function, allowing for channel-wise independent decay rates ($\odot$ denotes the Hadamard product). To guarantee a lossless gradient highway across spatial depth $L$, we incorporate a ReZero-inspired spatial residual correction:

$$ \boldsymbol{y}_t^{(l)} = \gamma \tilde{\boldsymbol{h}}_t^{(l)} + \text{Proj}(\boldsymbol{x}_t^{(l)}) $$

With the scalar $\gamma$ initialized to $0$, the spatial Jacobian at initialization rigorously equals the identity matrix $\boldsymbol{I}$, ensuring stable gradient flow during the early stages of training.

### 1.2 Low-Rank Markovian State Routing (LSR)

Constructing a Markov transition matrix directly in $\mathbb{R}^d$ space imposes $\mathcal{O}(d^2)$ overhead. To resolve this, we project the state into a bottleneck space $r \ll d$:

$$ \boldsymbol{q}_t = \boldsymbol{W}_Q \boldsymbol{h}_t, \quad \boldsymbol{k}_t = \boldsymbol{W}_K \boldsymbol{h}_t, \quad \boldsymbol{v}_t = \boldsymbol{W}_V \boldsymbol{h}_t \quad \in \mathbb{R}^r $$

The Markovian transition matrix $\boldsymbol{M}_t$ is constructed using the scaled dot-product form, normalized across columns to form a valid Right Stochastic Matrix:

$$ \boldsymbol{M}_t = \text{Softmax}\left( \frac{\boldsymbol{q}_t \boldsymbol{k}_t^\top}{\sqrt{r}}, \dim=-1 \right) $$

The flux is routed through this bottleneck and projected back to $\mathbb{R}^d$:

$$ \tilde{\boldsymbol{h}}_t^{(l)} = \boldsymbol{W}_O (\boldsymbol{M}_t \boldsymbol{v}_t) $$

### 1.3 Theorem 1: Emergence of Markovian Gradients

A key contribution of FluxProp is mathematically proving that routing in the forward pass intrinsically generates Markov-weighted gradients during Backpropagation Through Time (BPTT).

The analytic gradient with respect to the pre-routed state $\boldsymbol{h}_t$ rigorously converges to:

$$ \nabla_{\boldsymbol{h}_t} \mathcal{L} = \underbrace{\boldsymbol{W}_V^\top \boldsymbol{M}_t^\top \boldsymbol{W}_O^\top (\nabla_{\tilde{\boldsymbol{h}}_t} \mathcal{L})}_{\text{Main Term: Markovian Transpose Weighting}} + \underbrace{\boldsymbol{J}_{\text{Softmax}}(\boldsymbol{h}_t)^\top (\nabla_{\tilde{\boldsymbol{h}}_t} \mathcal{L})}_{\text{Correction Term: Probabilistic Manifold Reshaping}} $$

The emergence of the $\boldsymbol{M}_t^\top$ term demonstrates an elegant system duality: the Markovian transition probabilities naturally re-weight the backward gradient flow inversely, mathematically rendering heuristic backward hooks obsolete.

-----

## 2\. Training Strategy: Layer-wise Staged Pre-training

To maintain stability across the non-linear Jacobian flow, FluxProp utilizes a staged training pipeline:

  * **Layer-wise Solidification:** Freeze all layers except the target layer (starting from Layer 0) to solidify basic feature extraction.
  * **Progressive Unfreezing:** Gradually introduce higher layers into the training loop, inheriting previous knowledge.
  * **Full Fusion:** Finally, unfreeze all parameters for global optimization at a low learning rate to achieve system-wide resonance.

<br>
<br>

-----

