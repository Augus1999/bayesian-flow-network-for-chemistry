# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Define Bayesian Flow Network for Chemistry (ChemBFN) model.
"""
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax
from typing_extensions import Self


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class RoPE(nn.Module):
    def __init__(self, channel: int = 512, num_head: int = 8) -> None:
        """
        Rotary position embedding block with XPOS method.

        :param channel: hidden layer features
        :param num_head: number of heads
        """
        super().__init__()
        d = channel // num_head
        assert d % 2 == 0
        self.channel = channel
        i = torch.arange(0, d, 2)[None, :] / d
        theta_half = torch.pow(10000, -i)
        zeta_half = (i + 0.4) / 1.4
        theta, zeta = torch.zeros((1, d)), torch.zeros((1, d))
        theta[:, 0::2] = theta_half
        theta[:, 1::2] = theta_half
        zeta[:, 0::2] = zeta_half
        zeta[:, 1::2] = zeta_half
        self.register_buffer("theta", theta)
        self.register_buffer("zeta", zeta)

    def forward(self, size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param size: maximum length of sequence in the batch
        :return: cos part of position encoding;  shape: (1, 1, n_t, n_h)
                 sin part of position encoding;  shape: (1, 1, n_t, n_h)
                 scaling coefficients;           shape: (1, 1, n_t, n_h)
        """
        pos = torch.arange(size, device=self.theta.device)[:, None]
        cos, sin = torch.cos(pos * self.theta), torch.sin(pos * self.theta)
        zeta = torch.pow(self.zeta, pos / self.channel)
        return cos[None, None, ...], sin[None, None, ...], zeta[None, None, ...]


class Attention(nn.Module):
    def __init__(self, channel: int = 512, num_head: int = 8) -> None:
        """
        Multi-head self-attention block.

        :param channel: hidden layer features
        :param num_head: number of heads
        """
        super().__init__()
        assert channel % num_head == 0
        self.d = channel // num_head  # head dimension
        self.nh = num_head  # number of heads
        self.tp = (2 * self.d) ** 0.5  # attention temperature
        self.qkv = nn.Linear(channel, channel * 3)

    @staticmethod
    def _rotate(
        q: Tensor, k: Tensor, pe: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        q_rotate, k_rotate = torch.zeros_like(q), torch.zeros_like(k)
        q_rotate[..., 0::2] = -q[..., 1::2]
        q_rotate[..., 1::2] = q[..., 0::2]
        q = (q * pe[0] + q_rotate * pe[1]) * pe[2]
        k_rotate[..., 0::2] = -k[..., 1::2]
        k_rotate[..., 1::2] = k[..., 0::2]
        k = (k * pe[0] + k_rotate * pe[1]) / pe[2]
        return q, k

    def forward(
        self, x: Tensor, pe: Tuple[Tensor, Tensor, Tensor], mask: Optional[Tensor]
    ) -> Tensor:
        """
        :param x: output tensor;       shape: (n_b, n_t, n_f)
        :param pe: position encoding;  shape: (1, 1, n_t, n_h) * 3
        :param mask: attention mask;   shape: (1, n_b, n_t, n_t)
        :return: attentioned output;   shape: (n_b, n_t, n_f)
        """
        n_b, n_a, _ = x.shape
        split = (n_b, n_a, self.nh, self.d)
        shape = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(split).permute(2, 0, 1, 3).contiguous()
        k = k.view(split).permute(2, 0, 1, 3).contiguous()
        v = v.view(split).permute(2, 0, 1, 3).contiguous()
        q, k = self._rotate(q, k, pe)  # position embedding
        k_t = k.transpose(-2, -1)
        if mask is not None:
            alpha = softmax((q @ k_t / self.tp).masked_fill_(mask, -torch.inf), -1)
        else:
            alpha = softmax(q @ k_t / self.tp, -1)
        atten_out = (alpha @ v).permute(1, 2, 0, 3).contiguous().view(shape)
        return atten_out


class TransformerLayer(nn.Module):
    def __init__(
        self, channel: int = 512, num_head: int = 8, dropout: float = 0.01
    ) -> None:
        """
        Transfomer layer block.

        :param channel: hidden layer features
        :param num_head: number of attention heads
        :param dropout: dropout frequency
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(channel, 1e-6, False)
        self.attention = Attention(channel, num_head)
        self.norm2 = nn.LayerNorm(channel, 1e-6, False)
        self.ffn = nn.Sequential(
            nn.Linear(channel, channel * 4),
            nn.SELU(),
            nn.Linear(channel * 4, channel),
            nn.Dropout(dropout),
        )
        self.adaln_modulation = nn.Sequential(
            nn.SELU(), nn.Linear(channel, 6 * channel)
        )
        # zero-out adaLN layer
        nn.init.constant_(self.adaln_modulation[1].weight, 0)
        nn.init.constant_(self.adaln_modulation[1].bias, 0)

    def forward(
        self,
        x: Tensor,
        pe: Tuple[Tensor, Tensor, Tensor],
        c: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """
        :param x: input tensor;        shape: (n_b, n_t, n_f)
        :param pe: position encoding;  shape: (1, 1, n_t, n_h) * 3
        :param c: conditioning;        shape: (n_b, 1, n_f)
        :param mask: attention mask;   shape: (1, n_b, n_t, n_t)
        :return: output tensor;        shape: (n_b, n_t, n_f)
        """
        c = self.adaln_modulation(c)
        shift, scale, gate, shift_ffn, scale_ffn, gate_ffn = c.chunk(6, -1)
        x = x + gate * self.attention(modulate(self.norm1(x), shift, scale), pe, mask)
        x = x + gate_ffn * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        return x


class FinalLayer(nn.Module):
    def __init__(self, num_vocab: int, channel: int = 512) -> None:
        """
        The final layer of model.

        :param num_vocab: number of vocabulary
        :param channel: hidden layer features
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(channel, 1e-6, False)
        self.linear = nn.Linear(channel, num_vocab)
        self.adaln_modulation = nn.Sequential(
            nn.SELU(), nn.Linear(channel, 2 * channel)
        )
        # zero-out this layer
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor, return_logits: bool = True) -> Tensor:
        """
        :param x: input tensor;                 shape: (n_b, n_t, n_f)
        :param c: conditioning;                 shape: (n_b, 1, n_f)
        :param return_logits: whether to return unnormalised output logits
        :return: output logits (unnormalised);  shape: (n_b, n_t, n_vocab)
                 or first latent vector;        shape: (n_b, n_f)
        """
        shift, scale = self.adaln_modulation(c).chunk(2, -1)
        x = modulate(self.norm_final(x), shift, scale)
        if return_logits:
            return self.linear(x)
        return x[::, 0]


class ChemBFN(nn.Module):
    def __init__(
        self,
        num_vocab: int,
        channel: int = 512,
        num_layer: int = 12,
        num_head: int = 8,
        dropout: float = 0.01,
    ) -> None:
        r"""
        Bayesian Flow Network for Chemistry model representation.

        :param num_vocab: number of vocabulary
        :param channel: hidden layer features
        :param num_layer: number of transformer layers
        :param num_head: number of heads
        :param dropout: dropout frequency
        """
        super().__init__()
        self.K = num_vocab
        self.embedding = nn.Linear(num_vocab, channel)
        self.time_embed = nn.Sequential(
            nn.Linear(1, channel // 2), nn.SELU(), nn.Linear(channel // 2, channel)
        )
        self.position = RoPE(channel, num_head)
        self.encoder_layers = nn.ModuleList(
            [TransformerLayer(channel, num_head, dropout) for _ in range(num_layer)]
        )
        self.final_layer = FinalLayer(num_vocab, channel)
        self.register_buffer("beta", torch.scalar_tensor(20.4054 / self.K))
        self.hparam = dict(
            num_vocab=num_vocab,
            channel=channel,
            num_layer=num_layer,
            num_head=num_head,
            dropout=dropout,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param x: input probabilities;                       shape: (n_b, n_t, n_vocab)
        :param t: time;                                      shape: (n_b, 1)
        :param mask: input mask;                             shape: (n_b, n_t, 1)
        :param y: conditioning vector;                       shape: (n_b, 1, n_f)
        :return: probability distribution (before softmax);  shape: (n_b, n_t, n_vocab)
                 or first latent vector;                     shape: (n_b, n_f)
        """
        c = self.time_embed(t)[:, None, :]
        if y is not None:
            c += y
        pe = self.position(x.shape[1])
        x = self.embedding(x)
        if mask is not None:
            mask = mask.transpose(-2, -1).repeat(1, x.shape[1], 1)[None, ...] == 0
        for layer in self.encoder_layers:
            x = layer(x, pe, c, mask)
        return self.final_layer(x, c, mask is None)

    def calc_beta(self, t: Tensor) -> Tensor:
        r"""
        Calculate beta(t) value.

        \begin{equation}
            \beta(t) = %
            -\frac{4\ln{(1 - t + te^{-\frac{K}{4}\beta(1)})}}{K}
        \end{equation}

        :param t: continuous time in [0, 1];   shape: (n_b, 1)
        :return beta(t);                       shape: (n_b, 1)
        """
        return -4 * (1 - t + t * (-self.K * self.beta / 4).exp()).log() / self.K

    def calc_discrete_alpha(self, t1: Tensor, t2: Tensor) -> Tensor:
        r"""
        Calculate alpha(i) value.

        $\alpha(i) = \bate(t_{i}) - \beta(t_{i - 1})$

        :param t1: discrete time (i - 1) / n;  shape: (n_b, 1)
        :param t2: discrete time i / n;        shape: (n_b, 1)
        :return alpha(i);                      shape: (n_b, 1)
        """
        # assert t2 > t1
        return self.calc_beta(t2) - self.calc_beta(t1)

    def calc_cts_alpha(self, t: Tensor) -> Tensor:
        r"""
        Calculate alpha(t) / 2 value.

        \begin{equation}
            \alpha(t) = %
            \frac{d\beta(t)}{dt} = %
            \frac{4}{K}%
            \frac{1 - e^{-\frac{K}{4}\beta(1)}}%
            {1 - t + te^{-\frac{K}{4}\beta(1)}}
        \end{equation}

        :param t: continuous time in [0, 1];  shape: (n_b, 1)
        :return alpha(t);                     shape: (n_b, 1)
        """
        a = 1 - (-self.K * self.beta / 4).exp()
        b = 1 - t + t * (-self.K * self.beta / 4).exp()
        return 2 * a / b / self.K

    def discrete_output_distribution(
        self, theta: Tensor, t: Tensor, y: Optional[Tensor]
    ) -> Tensor:
        """
        :param theta: input distribution;     shape: (n_b, n_t, n_vocab)
        :param t: continuous time in [0, 1];  shape: (n_b, 1)
        :param y: conditioning vector;        shape: (n_b, 1, n_f)
        :return: output distribution;         shape: (n_b, n_t, n_vocab)
        """
        theta = 2 * theta - 1  # rescale to [-1, 1]
        return softmax(self.forward(theta, t, None, y), -1)

    def cts_loss(self, x: Tensor, t: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Compute continuous-time loss.

        :param x: target data;                shape: (n_b, n_t)
        :param t: continuous time in [0, 1);  shape: (n_b, 1)
        :param y: conditioning vector;        shape: (n_b, 1, n_f)
        :return: continuous-time loss;        shape: ()
        """
        beta = self.calc_beta(t)[..., None]  # shape: (n_b, 1, 1)
        e_x = nn.functional.one_hot(x, self.K).float()
        mu = beta * (self.K * e_x - 1)
        sigma = (beta * self.K).sqrt()
        theta = softmax(mu + sigma * torch.randn_like(mu), -1)
        e_hat = self.discrete_output_distribution(theta, t, y)
        cts_loss = self.K * (e_x - e_hat).pow(2) * self.calc_cts_alpha(t)[..., None]
        return cts_loss.mean()

    @torch.inference_mode()
    def reconstruction_loss(self, x: Tensor, t: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Compute reconstruction loss.

        :param x: target data;                shape: (n_b, n_t)
        :param t: continuous time in [0, 1];  shape: (n_b, 1)
        :param y: conditioning vector;        shape: (n_b, 1, n_f)
        :return: reconstruction loss;         shape: ()
        """
        beta = self.calc_beta(t)[..., None]
        mu = beta * (self.K * nn.functional.one_hot(x, self.K).float() - 1)
        sigma = (beta * self.K).sqrt()
        theta = softmax(mu + sigma * torch.randn_like(mu), -1)
        logits = self.forward(2 * theta - 1, t, None, y)
        # compute negative log probability
        x, logits = torch.broadcast_tensors(x[..., None], logits)
        return (-logits.gather(-1, x[..., :1]).squeeze(-1)).mean()

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        sequence_size: int,
        y: Optional[Tensor],
        sample_step: int = 1000,
    ) -> Tensor:
        """
        Sample from a piror distribution.

        :param batch_size: batch size
        :param sequence_size: max sequence length
        :param y: conditioning vector;      shape: (n_b, 1, n_f)
        :param sample_step: number of sampling steps
        :return: probability distribution;  shape: (n_b, n_t, n_vocab)
        """
        self.eval()
        theta = (
            torch.ones((batch_size, sequence_size, self.K), device=self.beta.device)
            / self.K
        )
        for i in torch.linspace(1, sample_step, sample_step, device=self.beta.device):
            t = (i - 1).view(1, 1).repeat(batch_size, 1) / sample_step
            p = self.discrete_output_distribution(theta, t, y)
            alpha = self.calc_discrete_alpha(t, t + 1 / sample_step)[..., None]
            e_k = nn.functional.one_hot(torch.argmax(p, -1), self.K).float()
            mu = alpha * (self.K * e_k - 1)
            sigma = (alpha * self.K).sqrt()
            theta = (mu + sigma * torch.randn_like(mu)).exp() * theta
            theta = theta / theta.sum(-1, True)
        t_final = torch.ones((batch_size, 1), device=self.beta.device)
        p = self.discrete_output_distribution(theta, t_final, y)
        return torch.argmax(p, -1)

    def inference(self, x: Tensor, mlp: nn.Module) -> Tensor:
        """
        Predict from SMILES tokens.

        :param x: input tokens;  shape: (n_b, n_t)
        :param mlp: MLP module
        :return: output values;  shape: (n_b, n_task)
        """
        t = torch.ones((x.shape[0], 1), device=x.device)
        mask = (x != 0).float()[..., None]
        theta = 2 * torch.nn.functional.one_hot(x, self.K).float() - 1
        z = self.forward(theta, t, mask, None)
        return mlp.forward(z)

    @classmethod
    def from_checkpoint(cls, ckpt: str, strict: bool = True) -> Self:
        """
        Load model weight from a checkpoint.

        :param ckpt: checkpoint file
        :param strict: whether to strictly match `state_dict`
        :return: Bayesian Flow Network for Chemistry model
        """
        with open(ckpt, "rb") as f:
            state = torch.load(f, "cpu")
        nn, hparam = state["nn"], state["hparam"]
        model = ChemBFN(
            hparam["num_vocab"],
            hparam["channel"],
            hparam["num_layer"],
            hparam["num_head"],
            hparam["dropout"],
        )
        model.load_state_dict(nn, strict)
        return model


class MLP(nn.Module):
    def __init__(self, size: List[int]) -> None:
        """
        MLP module.

        :param size: hidden feature sizes\n
        e.g.
        ```python
        mlp = MLP(size=[512, 64, 1])
        ```
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(i, size[key + 1]) for key, i in enumerate(size[:-2])]
        )
        self.layers.append(nn.Linear(size[-2], size[-1]))
        self.hparam = dict(size=size)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = torch.selu(layer(x))
        return self.layers[-1](x)

    @classmethod
    def from_checkpoint(cls, ckpt: str, strict: bool = True) -> Self:
        """
        load model weight from a checkpoint.

        :param ckpt: checkpoint file
        :param strict: whether to strictly match `state_dict`
        :return: MLP
        """
        with open(ckpt, "rb") as f:
            state = torch.load(f, "cpu")
        nn, hparam = state["nn"], state["hparam"]
        model = MLP(hparam["size"])
        model.load_state_dict(nn, strict)
        return model


if __name__ == "__main__":
    ...
