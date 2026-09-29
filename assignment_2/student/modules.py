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
        # return x @ self.W.T
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

    def forward(self, x: torch.Tensor):
        pass