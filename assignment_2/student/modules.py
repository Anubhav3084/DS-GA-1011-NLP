import math
import torch
import torch.nn as nn
from einops import einsum

class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(
            torch.empty(self.out_features, self.in_features, device=device, dtype=dtype),
            requires_grad=True
        )

        self.reset_params()

    def reset_params(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(
            self.W,
            mean=0.0, 
            std=std, 
            a=-3.0 * std, 
            b=3.0 * std
        )

    def forward(self, x: torch.Tensor):
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None, requires_grad=True):
        super(EmbeddingLayer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.E = nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype),
            requires_grad=requires_grad
        )

        self.reset_params()

    def reset_params(self):
        nn.init.trunc_normal_(
            self.E,
            mean=0.0,
            std=1.0,
            a=-3,
            b=3
        )

    def forward(self, x: torch.Tensor):
        return self.E[x]

class RMSNormLayer(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNormLayer, self).__init__()

        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype),
            requires_grad=True
        )

        self.reset_params()

    def reset_params(self):
        nn.init.ones_(self.g)

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x_float = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps)
        rmsnorm = (x_float / rms) * self.g.to(torch.float32)
        return rmsnorm.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = LinearLayer(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.W2 = LinearLayer(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.W3 = LinearLayer(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def SiLU(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        silg_x_W1 = self.SiLU(self.W1(x))
        x_W3 = self.W3(x)
        element_wise_mult = torch.mul(silg_x_W1, x_W3)
        return self.W2(element_wise_mult)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RoPE, self).__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        self.register_buffer("cos_buffer", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_buffer", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos_sliced = self.cos_buffer[token_positions].to(dtype=x.dtype)
        sin_sliced = self.sin_buffer[token_positions].to(dtype=x.dtype)

        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]

        out0 = x0 * cos_sliced - x1 * sin_sliced
        out1 = x0 * cos_sliced + x1 * sin_sliced

        out_combined = torch.stack([out0, out1], dim=-1)
        return out_combined.reshape(x.shape)

def SoftmaxLayer(in_features: torch.Tensor, dim: int):
    max_val_in_dim = torch.max(in_features, dim=dim, keepdim=True).values
    adjusted_values = in_features - max_val_in_dim
    exp_values = torch.exp(adjusted_values)
    sum_values = torch.sum(exp_values, dim=dim, keepdim=True)
    return exp_values / sum_values

def AttentionLayer(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None, dim=None):
    d_k = Q.shape[-1]
    Q_K_T = einsum(Q, K, " ... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    masked_Q_K_T = torch.where(mask, Q_K_T, float('-inf'))
    probs = SoftmaxLayer(masked_Q_K_T, dim=-1)
    return einsum(probs, V, " ... n m, ... m d_v -> ... n d_v")