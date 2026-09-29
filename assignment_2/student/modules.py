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