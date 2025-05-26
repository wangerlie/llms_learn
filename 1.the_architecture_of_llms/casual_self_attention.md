# Self-Attention Mechanism: Shape Transformations Guide

This document provides a detailed walkthrough of the self-attention mechanism with explicit tensor shape transformations, covering both single-head and multi-head scenarios.

## Table of Contents
- [Notation](#notation)
- [Single-Head Self-Attention](#single-head-self-attention)
- [Multi-Head Self-Attention](#multi-head-self-attention)
- [Causal Masking](#causal-masking)
- [Complete Example](#complete-example)

## Notation

| Symbol | Description |
|--------|-------------|
| `B` | Batch size |
| `T` | Sequence length |
| `C` | Embedding dimension (n_embd) |
| `H` | Number of attention heads |
| `hs` | Head size (C // H) |

---

## Single-Head Self-Attention

### Step 1: Input Preparation
```
Input tensor x: [B, T, C]
Example: [2, 4, 512]  # 2 sequences, 4 tokens each, 512-dim embeddings
```

### Step 2: Linear Projections
Two approaches for creating Q, K, V:

#### Approach A: Single Linear Layer
```
Linear layer: [C, 3*C] → [512, 1536]
x @ W_qkv → qkv: [B, T, 3*C] → [2, 4, 1536]

Split operation:
qkv.split(C, dim=-1) → q, k, v: each [B, T, C] → [2, 4, 512]
```

#### Approach B: Three Separate Linear Layers
```
W_q: [C, C] → [512, 512]    q = x @ W_q → [B, T, C] → [2, 4, 512]
W_k: [C, C] → [512, 512]    k = x @ W_k → [B, T, C] → [2, 4, 512]
W_v: [C, C] → [512, 512]    v = x @ W_v → [B, T, C] → [2, 4, 512]
```

### Step 3: Attention Score Calculation
```
Transpose k: k.transpose(-2, -1) → [B, C, T] → [2, 512, 4]

Matrix multiplication:
q @ k^T → attention_scores: [B, T, T] → [2, 4, 4]

Scaling:
attention_scores = attention_scores * (1.0 / sqrt(C))
                 = attention_scores * (1.0 / sqrt(512))
                 = attention_scores * 0.044
```

### Step 4: Attention Weight Matrix Structure
```
For sequence length T=4, attention matrix looks like:
    k1   k2   k3   k4
q1 [a11  a12  a13  a14]
q2 [a21  a22  a23  a24]  
q3 [a31  a32  a33  a34]
q4 [a41  a42  a43  a44]

Each aij represents: "How much should token i attend to token j?"
```

### Step 5: Softmax Normalization
```
Softmax along last dimension (dim=-1):
attention_weights = softmax(attention_scores, dim=-1)

Each row now sums to 1.0:
a11 + a12 + a13 + a14 = 1.0
a21 + a22 + a23 + a24 = 1.0
...
```

### Step 6: Value Aggregation
```
Final output:
y = attention_weights @ v
Shape: [B, T, T] @ [B, T, C] → [B, T, C] → [2, 4, 512]
```

---

## Multi-Head Self-Attention

### Step 1: Initial Setup
```
Input x: [B, T, C] → [2, 4, 512]
Number of heads H = 8
Head size hs = C // H = 512 // 8 = 64
```

### Step 2: Q, K, V Projections (Same as Single-Head)
```
q, k, v: each [B, T, C] → [2, 4, 512]
```

### Step 3: Reshape for Multi-Head
```
Original: q: [B, T, C] → [2, 4, 512]

Reshape to separate heads:
q = q.view(B, T, H, hs) → [2, 4, 8, 64]

Transpose for parallel processing:
q = q.transpose(1, 2) → [2, 8, 4, 64]
                       → [B, H, T, hs]

Similarly for k and v:
k: [B, H, T, hs] → [2, 8, 4, 64]
v: [B, H, T, hs] → [2, 8, 4, 64]
```

### Step 4: Parallel Attention Computation
```
Transpose k for matrix multiplication:
k.transpose(-2, -1): [B, H, hs, T] → [2, 8, 64, 4]

Attention scores:
q @ k^T: [B, H, T, T] → [2, 8, 4, 4]

Scaling:
scores = scores * (1.0 / sqrt(hs)) = scores * (1.0 / sqrt(64)) = scores * 0.125
```

### Step 5: Multi-Head Attention Matrix Structure
```
For H=8 heads, we now have 8 parallel attention matrices:

Head 1: [4x4]  Head 2: [4x4]  ...  Head 8: [4x4]

Each head focuses on different aspects of relationships between tokens.
```

### Step 6: Softmax and Value Aggregation
```
Softmax: [B, H, T, T] → [2, 8, 4, 4] (each head normalized independently)
Output: attention_weights @ v → [B, H, T, hs] → [2, 8, 4, 64]
```

### Step 7: Concatenate Heads
```
Transpose back: y.transpose(1, 2) → [B, T, H, hs] → [2, 4, 8, 64]

Concatenate heads:
y = y.contiguous().view(B, T, C) → [2, 4, 512]

Final shape matches input: [B, T, C]
```

---

## Causal Masking

### Mask Creation
```
For sequence length T=4:
mask = torch.tril(torch.ones(4, 4))

Mask matrix:
[1  0  0  0]
[1  1  0  0]
[1  1  1  0]
[1  1  1  1]

Broadcast to: [1, 1, T, T] for all batches and heads
```

### Mask Application
```
Before masking (example scores):
[ 0.2  0.3  0.1  0.4]
[ 0.5  0.2  0.1  0.2]
[ 0.3  0.4  0.2  0.1]
[ 0.1  0.3  0.4  0.2]

After masked_fill(mask == 0, float('-inf')):
[ 0.2  -inf  -inf  -inf]
[ 0.5   0.2  -inf  -inf]
[ 0.3   0.4   0.2  -inf]
[ 0.1   0.3   0.4   0.2]

After softmax:
[1.0   0.0   0.0   0.0]  # Token 0 only sees itself
[0.73  0.27  0.0   0.0]  # Token 1 sees tokens 0-1
[0.25  0.40  0.35  0.0]  # Token 2 sees tokens 0-2
[0.09  0.24  0.40  0.27] # Token 3 sees tokens 0-3
```

---

## Complete Example

### Input
```python
batch_size = 2
seq_len = 4
embed_dim = 512
n_heads = 8

x = torch.randn(2, 4, 512)  # [B, T, C]
```

### Shape Flow Summary
```
1. Input:           x: [2, 4, 512]
2. QKV projection:  q,k,v: each [2, 4, 512]
3. Multi-head reshape: q,k,v: each [2, 8, 4, 64]
4. Attention scores:   [2, 8, 4, 4]
5. After softmax:      [2, 8, 4, 4]
6. Attend to values:   [2, 8, 4, 64]
7. Concatenate heads:  [2, 4, 512]
8. Output projection:  [2, 4, 512]
```

### Key Insights
- **Parallel Processing**: All heads compute attention simultaneously
- **Shape Consistency**: Input and output shapes are identical
- **Information Flow**: Each position aggregates information from previous positions
- **Efficiency**: Single matrix operations handle multiple heads at once

---

## Mathematical Summary

The complete self-attention operation can be expressed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- `Q, K, V` are query, key, value matrices
- `d_k` is the key dimension (head_size)
- Masking is applied before softmax for causal attention
- Multiple heads process different representation subspaces in parallel