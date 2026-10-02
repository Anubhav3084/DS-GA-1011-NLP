import math
import torch
import torch.nn as nn
from einops import einsum, rearrange

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
        thetas = einsum(t, inv_freq, "i,j -> i j")

        self.register_buffer("cos_buffer", torch.cos(thetas), persistent=False)
        self.register_buffer("sin_buffer", torch.sin(thetas), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos_sliced = self.cos_buffer[token_positions].to(dtype=x.dtype)
        sin_sliced = self.sin_buffer[token_positions].to(dtype=x.dtype)

        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]

        out0 = x0 * cos_sliced - x1 * sin_sliced
        out1 = x0 * sin_sliced + x1 * cos_sliced

        out_combined = torch.stack([out0, out1], dim=-1)
        return out_combined.reshape(x.shape)

def SoftmaxLayer(in_features: torch.Tensor, dim: int):
    max_val_in_dim = torch.max(in_features, dim=dim, keepdim=True).values
    adjusted_values = in_features - max_val_in_dim
    exp_values = torch.exp(adjusted_values)
    sum_values = torch.sum(exp_values, dim=dim, keepdim=True)
    return exp_values / sum_values

def AttentionLayer(Query: torch.Tensor, Key: torch.Tensor, Value: torch.Tensor, mask=None, dim=None):
    d_k = Query.shape[-1]
    Q_K_T = einsum(Query, Key, " ... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    if mask is not None: masked_Q_K_T = torch.where(mask, Q_K_T, float('-inf'))
    else: masked_Q_K_T = Q_K_T
    probs = SoftmaxLayer(masked_Q_K_T, dim=-1)
    return einsum(probs, Value, " ... n m, ... m d_v -> ... n d_v")

class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = None, device=None, dtype=None):
        super(MultiHeadSelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = self.d_model // self.h
        self.d_v = self.d_model // self.h
        self.max_seq_len = max_seq_len
        self.device = device

        self.Query = LinearLayer(in_features=self.d_model, out_features=self.d_k * self.h, device=device, dtype=dtype)
        self.Key = LinearLayer(in_features=self.d_model, out_features=self.d_k * self.h, device=device, dtype=dtype)
        self.Value = LinearLayer(in_features=self.d_model, out_features=self.d_v * self.h, device=device, dtype=dtype)

        self.Output = LinearLayer(in_features=self.d_v * self.h, out_features=self.d_model, device=device, dtype=dtype)

        if theta is not None: self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=self.max_seq_len, device=device)

    def forward(self, x, token_positions=None):
        Q_out = self.Query(x)
        K_out = self.Key(x)
        V_out = self.Value(x)
        Q_out = rearrange(Q_out, " ... seq (heads d_k) -> ... heads seq d_k", heads=self.h)
        K_out = rearrange(K_out, " ... seq (heads d_k) -> ... heads seq d_k", heads=self.h)
        V_out = rearrange(V_out, " ... seq (heads d_v) -> ... heads seq d_v", heads=self.h)

        if token_positions is not None:
            Q_out = self.rope(Q_out, token_positions)
            K_out = self.rope(K_out, token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))
        multihead_attention = AttentionLayer(Q_out, K_out, V_out, mask=mask)
        multihead_attention = rearrange(multihead_attention, " ... heads seq d_v -> ... seq (heads d_v)", heads=self.h)
        return self.Output(multihead_attention)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super(TransformerBlock, self)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # layer 1
        self.rmsnorm1 = RMSNormLayer(d_model=self.d_model, device=device, dtype=dtype)
        self.multihead = MultiHeadSelfAttentionLayer(
                            d_model=self.d_model,
                            num_heads=self.num_heads,
                            max_seq_len=self.max_seq_len,
                            theta=self.theta,
                            device=device,
                            dtype=dtype
                        )
        ## layer 2
        self.rmsnorm2 = RMSNormLayer(d_model=d_model, device=device)
        self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff, device=device, dtype=dtype)

    def forward(self, x):
        y = x + self.multihead(self.rmsnorm1(x))
        out = y + self.ffn(self.rmsnorm2(y))
        return out