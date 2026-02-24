# Bare-Metal Gemma 2: Mathematical Specification

This document provides a language-agnostic mathematical breakdown of the **Gemma 2 (2B)** inference engine. It focuses on the tensor operations, activation functions, and architectural quirks necessary for a "from-scratch" implementation (e.g., in Rust, Nim, or C++), without relying on deep learning frameworks.

## 1. Model Configuration (2B-IT)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | 256,128 | Vocabulary size |
| `d_model` | 2,304 | Hidden dimension (embedding dim) |
| `n_layers` | 26 | Number of transformer blocks |
| `n_heads` | 8 | Number of Query heads |
| `n_kv_heads` | 4 | Number of Key/Value heads (GQA) |
| `head_dim` | 256 | Dimension per head (Note: $n_{heads} \cdot d_{head} = 2048 \neq d_{model}$) |
| `hidden_dim` | 9,216 | MLP intermediate dimension (GeGLU) |
| `max_seq_len`| 8,192 | Maximum context length |
| `sliding_window`| 4,096 | Sliding window size for SWA layers |
| `soft_cap_attn`| 50.0 | Attention logit soft-capping value |
| `soft_cap_final`| 30.0 | Final logit soft-capping value |

## 2. Mathematical Components

### 2.1 RMSNorm (Root Mean Square Layer Normalization)
Gemma 2 applies RMSNorm at four points per layer: pre-attention, post-attention, pre-FFN, and post-FFN.

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma $$

*   **Note on Weights:** Depending on the export source, $\gamma$ might be stored as a "delta" weight. If the weights are small (centered around 0), use $(1 + \gamma)$. If they are centered around 1, use $\gamma$ directly.
*   **PyTorch Reference:** [`torch.nn.RMSNorm`](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
*   **Constant:** $\epsilon = 10^{-6}$

### 2.2 Rotary Positional Embeddings (RoPE)
RoPE encodes positional information by rotating pairs of dimensions in the $Q$ and $K$ vectors.

For a pair of components $(x_i, x_{i+d/2})$ at position $m$:
$$ \begin{pmatrix} x_i' \\ x_{i+d/2}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+d/2} \end{pmatrix} $$
where $\theta_i = 10000^{-2i/d}$.

*   **Reference:** [RoPE Paper (Arxiv)](https://arxiv.org/abs/2104.09864)

### 2.3 Grouped-Query Attention (GQA) with Soft-Capping
Gemma 2 2B uses GQA where 8 Query heads share 4 Key/Value heads (ratio 2:1).

1.  **Linear Projection:** $Q \in \mathbb{R}^{L \times 2048}, K \in \mathbb{R}^{L \times 1024}, V \in \mathbb{R}^{L \times 1024}$.
2.  **Logit Computation:** $A = \frac{QK^T}{\sqrt{d_{head}}}$.
3.  **Soft-Capping:**
    $$ A_{\text{capped}} = 50.0 \cdot \tanh\left(\frac{A}{50.0}\right) $$
4.  **Sliding Window Mask:** For layers $\{0, 2, 4, \dots\}$, a sliding window mask of size 4096 is applied in addition to the causal mask.
5.  **Softmax:** $\text{Attn} = \text{Softmax}(A_{\text{capped}} + \text{Mask})$.
6.  **Output:** $O = \text{Attn} \cdot V$.

*   **PyTorch Reference:** [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

### 2.4 Gated MLP (GeGLU)
The feed-forward network uses a gated architecture.

$$ \text{FFN}(x) = (\text{GELU}(xW_{\text{gate}}) \otimes (xW_{\text{up}}))W_{\text{down}} $$

*   **GELU Approximation:** $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$.
*   **PyTorch Reference:** [`torch.nn.functional.gelu`](https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html)

## 3. Forward Pass Sequence

For each token $t$ at position $m$:

1.  **Embedding Lookup:** $x_0 = \text{Lookup}(W_{emb}, t) \cdot \sqrt{d_{model}}$.
2.  **Transformer Layers (repeat 26 times):**
    *   $x_{norm} = \text{RMSNorm}_{pre\_attn}(x_i)$
    *   $q, k, v = \text{Project}(x_{norm})$
    *   $q, k = \text{RoPE}(q, k, m)$
    *   $attn = \text{GQA}(q, k, v, \text{mask}, \text{cap}=50)$
    *   $x_{mid} = x_i + \text{RMSNorm}_{post\_attn}(\text{Linear}_{out}(attn))$
    *   $x_{ffn\_norm} = \text{RMSNorm}_{pre\_ffn}(x_{mid})$
    *   $ffn = \text{GeGLU}(x_{ffn\_norm})$
    *   $x_{i+1} = x_{mid} + \text{RMSNorm}_{post\_ffn}(\text{Linear}_{down}(ffn))$
3.  **Final Norm:** $x_{final} = \text{RMSNorm}_{final}(x_{26})$.
4.  **Output Head:** $\text{logits} = x_{final} W_{emb}^T$.
5.  **Final Soft-Capping:** $\text{logits} = 30.0 \cdot \tanh(\text{logits} / 30.0)$.

## 4. Tensor Layout & Implementation Notes

Weights are stored in **Row-Major** order. For a linear layer $Y = XW^T$:
- If $W$ has shape $(\text{out}, \text{in})$, then $W[i, j]$ is at index `i * in + j`.
- Matrix multiplication: $Y_i = \sum_{j=0}^{\text{in}} X_j \cdot W_{i,j}$.

### KV Cache Logic
To handle sliding window attention in the KV cache:
- **Global Layers:** Keep all tokens up to `max_seq_len`.
- **SWA Layers:** Only the most recent 4096 tokens are strictly needed, though keeping all is mathematically equivalent if the mask is applied correctly.

### Memory Mapping
Zero-copy inference is achieved by mapping the binary weight file into memory. In Nim or Rust, this is handled via `mmap` / `memmap2`, treating the resulting pointer as a raw array of `f32`.

## 5. References
- [Google Gemma 2 Technical Report](https://arxiv.org/abs/2408.00118)
- [Hugging Face Gemma 2 Modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py)
- [Rotary Embeddings (RoPE) Explained](https://blog.eleuther.ai/rotary-embeddings/)
