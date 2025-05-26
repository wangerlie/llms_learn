import math
import torch
from torch import nn
# This file implements the casual self-attention mechanism used in GPT models.
# Details can be found in markdown file casual_self_attention.md
# and the original paper "Attention is All You Need" by Vaswani et al.
# https://arxiv.org/abs/1706.03762
class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super(CasualSelfAttention, self).__init__()
        self.n_head = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.attn = nn.Linear(self.n_embd, self.n_embd * 3, bias=False) # put Q, K, V together in one linear layer 
        self.proj = nn.Linear(self.n_embd,self.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # create a causal mask for self-attention

    def forward(self,x):
        B,T,C = x.size() # B: batch size, T: sequence length, C: embedding dimension
        q,k,v = self.attn(x).split(self.n_embd, dim=-1) # split the output of the linear layer into Q, K, V
        # turn q,k,v to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # calculate attention scores
        # change k to (B, n_head, head_size, T) for matrix multiplication
        # apply scaling to the attention scores to prevent large values
        attn_score = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) #(B, n_head, T, T)
        # apply the causal mask to the attention scores
        # 1 0 0
        # 1 1 0
        # 1 1 1
        attn_score = attn_score.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # the last dimension represents the keys that each query position can attend to.
        # q1k1, q1k2, q1k3
        # q2k1, q2k2, q2k3
        # q3k1, q3k2, q3k3
        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = self.attn_dropout(attn_score)
        y = attn_score @ v # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y)) # (B, T, C)
        return y

