# FluxProp-Stateful-Liquid-Markovian-Language-Model
FluxProp is an experimental, highly ambitious language model architecture. It breaks the traditional dichotomy between Transformers and standard RNNs by deeply 
###  Core Architecture & Theoretical Foundations

FluxProp is not just an engineering optimization; it is grounded in a rigorous joint spatio-temporal Jacobian analysis of continuous-time dynamics. The complete forward propagation strictly operates in $O(dr+r^{2})$ complexity per layer, formalized through three key mechanisms:

#### 1. Temporal Liquid Update & Spatial Residuals
At its core, the evolution of the hidden state over continuous time is governed by a nonlinear ODE. Discretized via the Forward Euler method, the temporal update for layer $l$ at step $t$ is:

$$
h_{t}^{(l)} = (1-\alpha)h_{t-1}^{(l)} + \tanh(W^{(l)}h_{t-1}^{(l)} + U^{(l)}x_{t}^{(l)} + b^{(l)})
$$

**The Spatial Vanishing Gradient Problem:** In a deep stacked architecture, the spatial Jacobian (Depth Flow) is $J_{space}^{(l)} = \text{diag}(f^{\prime}(\cdot))U^{(l)}$. Because $f^{\prime}(\cdot) \le 1$, the continuous multiplication $\prod_{i=1}^{L}J_{space}^{(i)}$ decays exponentially, neutralizing deep feature learning during Backpropagation Through Time (BPTT).

**Spatial Residual Correction:** To solve this, we explicitly inject an identity mapping into the spatial dimension:

$$
y_{t}^{(l)} = \tilde{h}_{t}^{(l)} + x_{t}^{(l)}
$$

Consequently, the spatial Jacobian becomes $J_{space}^{(l)} + I$. This identity matrix anchors the singular values near 1, mathematically guaranteeing a lossless gradient highway across spatial depth $L$.

#### 2. Low-Rank Markovian State Routing (LSR)
Constructing a Markov transition matrix directly on $h_{t} \in \mathbb{R}^{d}$ yields an unacceptable $O(d^{2})$ computational and memory overhead. To resolve this, we project the state into a lower-dimensional bottleneck space $r \ll d$:

$$
q_{t} = W_{Q}h_{t}, \quad k_{t} = W_{K}h_{t}, \quad v_{t} = W_{V}h_{t}
$$

We construct the low-rank transition matrix $M_{t} \in \mathbb{R}^{r \times r}$:

$$
M_{t} = \text{Softmax}\left(\frac{q_{t}k_{t}^{\top}}{\sqrt{r}}\right)
$$

The flux is routed through this bottleneck and projected back to $\mathbb{R}^{d}$ via an output projection $W_{O}$:

$$
\tilde{h}_{t} = W_{O}(M_{t}v_{t})
$$

This structural sparsification maintains Markovian properties while drastically reducing overhead.

#### 3. Theorem 1: Emergence of Markovian Gradients
A major theoretical contribution of FluxProp is proving that routing in the forward pass intrinsically generates the correct Markov-weighted gradients during BPTT. 

By applying the chain rule to the pre-routed state, the gradient rigorously converges to:

$$
\frac{\partial\mathcal{L}}{\partial h_{t}} = M_{t}^{\top}(\nabla_{\tilde{h}_{t}}\mathcal{L}) + Jacobian_{Softmax}(h_{t})^{\top}(\nabla_{\tilde{h}_{t}}\mathcal{L})
$$

The dominant term $M_{t}^{\top}\nabla_{\tilde{h}_{t}}\mathcal{L}$ demonstrates that the Markovian transition probability naturally re-weights the gradient flow, mathematically rendering heuristic backward hooks obsolete.
