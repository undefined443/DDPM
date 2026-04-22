import torch as th
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.ff = nn.Linear(input_dim, input_dim)
        self.act = nn.GELU()

    def forward(self, x: th.Tensor):
        return x + self.act(self.ff(self.ln(x)))


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model: int, scale: float = 1.0):
        super().__init__()
        self.d = d_model
        self.scale = scale
        i = self.d // 2
        freq = th.log(th.Tensor([10000.0])) / (i - 1)
        freq = th.exp(-freq * th.arange(i))
        self.freq = nn.Parameter(freq, requires_grad=False)

    def forward(self, pos: th.Tensor):
        pos = pos * self.scale
        angle = pos * self.freq.unsqueeze(0)
        emb = th.cat((th.sin(angle), th.cos(angle)), dim=-1)
        return emb


class ToyModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_emb_dim: int = 128,
        twoD_data: bool = True,
    ):
        super().__init__()
        self.twoD_data = twoD_data
        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        if twoD_data:  # heart
            self.input_emb1 = SinusoidalEmbedding(time_emb_dim, scale=25.0)
            self.input_emb2 = SinusoidalEmbedding(time_emb_dim, scale=25.0)
            concat_size = 2 * time_emb_dim + time_emb_dim  # 2d concat time
            self.input_dim = 2
        else:  # mnist
            concat_size = 28 * 28 + time_emb_dim  # mnist is 28*28
            self.input_dim = 28 * 28

        layers = [nn.Linear(concat_size, hidden_dim), nn.GELU()]
        for _ in range(num_layers):
            layers.append(ResBlock(hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, self.input_dim))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t = self.time_emb(t)
        if self.twoD_data:
            x_emb = self.input_emb1(x[:, 0].unsqueeze(-1))
            y_emb = self.input_emb2(x[:, 1].unsqueeze(-1))
            x = th.cat((x_emb, y_emb), dim=-1)
        x = th.cat((x, t), dim=-1)
        x = self.joint_mlp(x)
        return x
