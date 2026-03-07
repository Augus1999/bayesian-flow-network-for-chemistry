# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa Sueno)
"""
Geometry modules.
"""
from pathlib import Path
from typing import Tuple, Optional, Union, Self
import torch
from torch import nn, Tensor


class RBF(nn.Module):
    """
    RBF block.
    """

    def __init__(self, cell: float = 5.0, num_kernel: int = 64) -> None:
        """
        Gaussian-style RBF kernel.

        :param cell: unit cell length
        :param num_kernel: number of kernels
        :type cell: float
        :type num_kernel: int
        """
        super().__init__()
        self.register_buffer("cell", torch.tensor([cell]))
        self.register_buffer("num_kernel", torch.tensor([num_kernel]))
        offsets = torch.linspace((-self.cell).exp().item(), 1, num_kernel)
        offsets = offsets[None, None, None, :]
        coeff = ((1 - (-self.cell).exp()) / num_kernel).pow(-2) * torch.ones_like(
            offsets
        )
        self.offsets = nn.Parameter(offsets, requires_grad=True)
        self.coeff = nn.Parameter(coeff / 4, requires_grad=True)

    def forward(self, d: Tensor) -> Tensor:
        """
        :param d: distances;         shape: (1, n_a, k)
        :type d: torch.Tensor
        :return: distance features;  shape: (1, n_a, k, n_kernel)
        :rtype: torch.Tensor
        """
        return (-self.coeff * ((-d[..., None]).exp() - self.offsets).pow(2)).exp()


class CosinCutOff(nn.Module):
    """
    Continuous cutoff block.
    """

    def __init__(self, cutoff: float = 5.0) -> None:
        """
        Compute cosin-cutoff mask.

        :param cutoff: cutoff radius
        :type cutoff: float
        """
        super().__init__()
        self.register_buffer("cutoff", torch.tensor([cutoff]))

    def forward(self, d: Tensor) -> Tensor:
        """
        :param d: distances;   shape: (1, n_a, k)
        :type d: torch.Tensor
        :return: cutoff mask;  shape: (1, n_a, k)
        :rtype: torch.Tensor
        """
        cutoff = 0.5 * (torch.pi * d / self.cutoff).cos() + 0.5
        return cutoff.masked_fill_(d > self.cutoff, 0)


class Distance(nn.Module):
    """
    Distance block.
    """

    def __init__(self, max_neighbour: int = 15) -> None:
        """
        Compute pair-wise distances and normalised vectors.

        :param max_neighbour: maximum number of atom-nighbours
        :type max_neighbour: int
        """
        super().__init__()
        self.k = max_neighbour

    def forward(
        self, r: Tensor, batch_mask: Tensor, lattice: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param r: nuclear coordinates;    shape: (1, n_a, 3)
        :param batch_mask: batch mask;    shape: (1, n_a, n_a, 1)
        :param lattice: lattice vectors;  shape: (1, n_a, 3, 3)
        :type r: torch.Tensor
        :type batch_mask: torch.Tensor
        :type lattice: torch.Tensor | None
        :return: distances;     shape: (1, n_a, k) \n
                 edge vectors;  shape: (1, n_a, k, 3) \n
                 edge indices;  shape: (1, n_a, k)
        :rtype: tuple
        """
        n_a = r.shape[1]
        k = min(self.k, n_a - 1)
        vec = r[:, :, None, :] - r[:, None, :, :]
        vec.masked_fill_(batch_mask, torch.inf)  # mask the 'off-diagonal' elements
        loop_mask = torch.eye(n_a, device=r.device)[None, ...] == 0
        if lattice is not None:
            # compute distances under periodic boundary conditions
            r_shift1 = r + lattice[::, ::, 0]
            r_shift2 = r + lattice[::, ::, 1]
            r_shift3 = r + lattice[::, ::, 2]
            vec_shift1 = r[:, :, None, :] - r_shift1[:, None, :, :]
            vec_shift2 = r[:, :, None, :] - r_shift2[:, None, :, :]
            vec_shift3 = r[:, :, None, :] - r_shift3[:, None, :, :]
            vecs = torch.cat([vec, vec_shift1, vec_shift2, vec_shift3], dim=0)
            ds = torch.linalg.norm(vecs, 2, -1)
            d_min = torch.min(ds, dim=0)  # find min distances
            d, d_key = d_min.values[None, ...], d_min.indices
            vec = torch.gather(vecs, 0, d_key[None, :, :, None].repeat(1, 1, 1, 3))
            d_tril = torch.tril(d, -1)
            d_triu = torch.triu(d, 0).transpose_(-2, -1)
            d_tri = torch.cat([d_tril.unsqueeze(0), d_triu.unsqueeze(0)], 0)
            d_tri_min = torch.min(d_tri, dim=0)  # use symmetry
            d_tri, d_key = d_tri_min.values, d_tri_min.indices
            vec_tril = vec * (d_tril != 0).float().unsqueeze_(-1)
            vec_triu = (vec * (d_triu == 0).float().unsqueeze_(-1)).transpose(-2, -3)
            vec_tri = torch.cat([vec_tril, vec_triu], 0)
            vec = torch.gather(vec_tri, 0, d_key[..., None].repeat(1, 1, 1, 3))
            vec = vec - vec.transpose(-2, -3)
            vec = vec[loop_mask].view(1, n_a, n_a - 1, 3)  # remove 0 vectors
            d = (d_tri + d_tri.transpose(-2, -1))[loop_mask].view(1, n_a, n_a - 1)
        else:
            vec = vec[loop_mask].view(1, n_a, n_a - 1, 3)  # remove 0 vectors
            d = torch.linalg.norm(vec, 2, -1)
        edge = (-d).topk(k, dim=-1)
        d, idxs = -edge.values, edge.indices
        vec_idxs = idxs[..., None].repeat(1, 1, 1, 3)
        vec = vec.masked_fill(vec == torch.inf, 0).gather(dim=-2, index=vec_idxs)
        return d, vec, idxs


class Embedding(nn.Module):
    """
    Embedding block for EGNN.
    """

    def __init__(
        self, num_embed: int = 120, channel: int = 256, mol_channel: int = 512
    ) -> None:
        """
        Atomic embedding block.

        :param num_embed: number of embedded elements
        :param channel: hidden layer features
        :param mol_channel: molecular embedding features
        :type num_embed: int
        :type channel: int
        :type mol_channel: int
        """
        super().__init__()
        self.embed = nn.Embedding(num_embed, channel)
        self.linear = nn.Linear(2 * channel + mol_channel, channel)
        self.time_embed = nn.Sequential(
            nn.Linear(1, channel // 2), nn.SELU(), nn.Linear(channel // 2, channel)
        )

    def forward(self, z: Tensor, mol_embed: Tensor, batch: Tensor, t: Tensor) -> Tensor:
        """
        :param z: atomic numbers;               shape: (1, n_a)
        :param mol_embed: molecule embeddings;  shape: (n_b, n_embed)
        :param batch: batch mask;               shape: (n_b, n_a, 1)
        :param t: continuous time in [0, 1];    shape: (1, n_a, 1)
        :type z: torch.Tensor
        :type mol_embed: torch.Tensor
        :type batch: torch.Tensor
        :type t: torch.Tensor
        :return: atomic embeddings;             shape: (1, n_a, n_f)
        :rtype: torch.Tensor
        """
        n_b, n_a, _ = batch.shape
        x = self.embed(z)
        x = torch.cat(
            [x.repeat(n_b, 1, 1), mol_embed[:, None, :].repeat(1, n_a, 1)],
            -1,
        )
        x = (x * batch).sum(0, True)
        return self.linear(torch.cat([x, self.time_embed(t)], -1))


class CFConv(nn.Module):
    """
    Continuous filtering convolution block for EGNN.
    """

    def __init__(
        self,
        cutoff_radius: float = 5.0,
        num_kernel: int = 64,
        max_neighbour: int = 15,
        channel: int = 256,
    ) -> None:
        """
        Continuous filtering convolution block.

        :param cutoff_radius: cutoff radius
        :param num_kernel: number of RBF features
        :param max_neighbour: maximum number of atom-nighbours
        :param channel: hidden layer features
        :type cutoff_radius: float
        :type num_kernel: int
        :type max_neighbour: int
        :type channel: int
        """
        super().__init__()
        self.distance = Distance(max_neighbour)
        self.cutoff = CosinCutOff(cutoff_radius)
        self.rbf = RBF(cutoff_radius, num_kernel)
        self.dense = nn.Sequential(
            nn.Linear(num_kernel, channel, False),
            nn.SiLU(),
            nn.Linear(channel, channel, False),
            nn.SiLU(),
        )

    def forward(
        self, x: Tensor, r: Tensor, batch_mask: Tensor, lattice: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: atomic embedding;       shape: (1, n_a, n_f)
        :param r: nuclear coordinates;    shape: (1, n_a, 3)
        :param batch_mask: batch mask;    shape: (1, n_a, n_a, 1)
        :param lattice: lattice vectors;  shape: (1, n_a, 3, 3)
        :type x: torch.Tensor
        :type r: torch.Tensor
        :type batch_mask: torch.Tensor
        :type lattice: torch.Tensor | None
        :return: convoluted atomic embedding;  shape: (1, n_a, k, n_f) \n
                 edge vectors;                 shape: (1, n_a, k, 3)
        :rtype: tuple
        """
        d, vec, idx = self.distance(r, batch_mask, lattice)
        cutoff = self.cutoff(d)[..., None]
        e_ij = self.dense(self.rbf(d) * cutoff)
        x_j = x[0][idx]
        return e_ij * x_j, vec


class Interaction(nn.Module):
    """
    Interaction block of EGNN.
    """

    def __init__(
        self,
        cutoff_radius: float = 5.0,
        num_kernel: int = 64,
        max_neighbour: int = 15,
        channel: int = 256,
    ) -> None:
        """
        Interaction block.

        :param cutoff_radius: cutoff radius
        :param num_kernel: number of RBF features
        :param max_neighbour: maximum number of atom-nighbours
        :param channel: hidden layer features
        :type cutoff_radius: float
        :type num_kernel: int
        :type max_neighbour: int
        :type channel: int
        """
        super().__init__()
        self.linear = nn.Linear(channel, channel)
        self.cfconv = CFConv(cutoff_radius, num_kernel, max_neighbour, channel)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel),
            nn.SiLU(),
            nn.Linear(channel, channel + 1),
        )

    def forward(
        self, x: Tensor, r: Tensor, batch_mask: Tensor, lattice: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: atomic embedding;       shape: (1, n_a, n_f)
        :param r: nuclear coordinates;    shape: (1, n_a, 3)
        :param batch_mask: batch mask;    shape: (1, n_a, n_a, 1)
        :param lattice: lattice vectors;  shape: (1, n_a, 3, 3)
        :type x: torch.Tensor
        :type r: torch.Tensor
        :type batch_mask: torch.Tensor
        :type lattice: torch.Tensor | None
        :return: updated atomic embedding;     shape: (1, n_a, n_f) \n
                 updated nuclear coordinates;  shape: (1, n_a, 3)
        :rtype: tuple
        """
        n_f = x.shape[-1]
        v_ij, vec = self.cfconv(self.linear(x), r, batch_mask, lattice)
        v_ij, s_ij = torch.split(self.mlp(v_ij), [n_f, 1], -1)
        return x + v_ij.sum(-2), r + (vec * s_ij).sum(-2)


class EGNN(nn.Module):
    """
    EGNN block for comformer searching via BFN.
    """

    def __init__(
        self,
        num_embed: int = 120,
        channel: int = 256,
        mol_channel: int = 512,
        cutoff_radius: float = 5.0,
        num_kernel: int = 64,
        max_neighbour: int = 15,
        num_layer: int = 6,
    ) -> None:
        """
        Equivariant Graph Neural Network representation.

        :param num_embed: number of embedded elements
        :param channel: hidden layer features
        :param mol_channel: molecular embedding features
        :param cutoff_radius: cutoff radius
        :param num_kernel: number of RBF features
        :param max_neighbour: maximum number of atom-nighbours
        :param num_layer: number of Interaction blocks
        :type num_embed: int
        :type channel: int
        :type mol_channel: int
        :type cutoff_radius: float
        :type num_kernel: int
        :type max_neighbour: int
        :type num_layer: int
        """
        super().__init__()
        self.embed = Embedding(num_embed, channel, mol_channel)
        self.interaction_layers = nn.ModuleList(
            [
                Interaction(cutoff_radius, num_kernel, max_neighbour, channel)
                for _ in range(num_layer)
            ]
        )
        self.register_buffer("sigma", torch.scalar_tensor(1e-3))
        self.hparam = {
            "num_embed": num_embed,
            "channel": channel,
            "mol_channel": mol_channel,
            "cutoff_radius": cutoff_radius,
            "num_kernel": num_kernel,
            "max_neighbour": max_neighbour,
            "num_layer": num_layer,
        }

    def forward(
        self,
        z: Tensor,
        r: Tensor,
        batch: Tensor,
        mol_embed: Tensor,
        t: Tensor,
        lattice: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param z: atomic numbers;               shape: (1, n_a)
        :param r: nuclear coordinates;          shape: (1, n_a, 3)
        :param batch: batch mask;               shape: (n_b, n_a, 1)
        :param mol_embed: molecule embeddings;  shape: (n_b, n_embed)
        :param t: continuous time in [0, 1];    shape: (1, n_a, 1)
        :param lattice: lattice vectors;        shape: (n_b, 3, 3)
        :type z: torch.Tensor
        :type r: torch.Tensor
        :type batch: torch.Tensor
        :type mol_embed: torch.Tensor
        :type t: torch.Tensor
        :type lattice: torch.Tensor | None
        :return: new nuclear coordinates;        shape: (1, n_a, 3)
        :rtype: torch.Tensor
        """
        if lattice is not None:
            lattice = (lattice[:, None, :, :] * batch[..., None]).sum(0, True)
        _b = batch.squeeze(-1)
        batch_mask = (_b.transpose(-2, -1) @ _b == 0)[None, :, :, None]
        x = self.embed(z, mol_embed, batch, t)
        for layer in self.interaction_layers:
            x, r = layer(x, r, batch_mask, lattice)
        return r

    def cts_output_prediction(
        self,
        z: Tensor,
        mu: Tensor,
        batch: Tensor,
        mol_embed: Tensor,
        t: Tensor,
        gamma: Tensor,
        lattice: Optional[Tensor],
    ) -> Tensor:
        """
        :param z: atomic numbers;               shape: (1, n_a)
        :param mu: blured nuclear coordinates;  shape: (1, n_a, 3)
        :param batch: batch mask;               shape: (n_b, n_a, 1)
        :param mol_embed: molecule embeddings;  shape: (n_b, n_embed)
        :param t: continuous time in [0, 1];    shape: (1, n_a, 1)
        :param gamma: gamma parameters;         shape: (1, n_a, 1)
        :param lattice: lattice vectors;        shape: (n_b, 3, 3)
        :type z: torch.Tensor
        :type mu: torch.Tensor
        :type batch: torch.Tensor
        :type mol_embed: torch.Tensor
        :type t: torch.Tensor
        :type gamma: torch.Tensor
        :type lattice: torch.Tensor | None
        :return: predicted nuclear coordinates;  shape: (1, n_a, 3)
        :rtype: torch.Tensor
        """
        eps = self.forward(z, mu, batch, mol_embed, t, lattice)
        x_hat = torch.where(
            t.repeat(1, 1, 3) >= 1e-6,
            mu / gamma - ((1 - gamma) / gamma).sqrt() * eps,
            0,
        )
        return x_hat.clamp(-100, 100)

    def continuous_time_loss(
        self,
        z: Tensor,
        r: Tensor,
        batch: Tensor,
        mol_embed: Tensor,
        t: Tensor,
        lattice: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute continuous-time loss.

        :param z: atomic numbers;               shape: (1, n_a)
        :param r: nuclear coordinates;          shape: (1, n_a, 3)
        :param batch: batch mask;               shape: (n_b, n_a, 1)
        :param mol_embed: molecule embeddings;  shape: (n_b, n_embed)
        :param t: continuous time in [0, 1);    shape: (n_b, 1, 1)
        :param lattice: lattice vectors;        shape: (n_b, 3, 3)
        :type z: torch.Tensor
        :type r: torch.Tensor
        :type batch: torch.Tensor
        :type mol_embed: torch.Tensor
        :type t: torch.Tensor
        :type lattice: torch.Tensor | None
        :return: continuous time loss;          shape: ()
        :rtype: torch.Tensor
        """
        n_a = z.shape[-1]
        t = (t.repeat(1, n_a, 1) * batch).sum(0, True)
        gamma = 1 - (a := self.sigma.pow(2 * t))
        mu = gamma * r + (gamma * (1 - gamma)).sqrt() * torch.randn_like(r)
        x_hat = self.cts_output_prediction(z, mu, batch, mol_embed, t, gamma, lattice)
        loss = -self.sigma.log() * (r - x_hat).pow(2) / a
        return loss.mean()

    @torch.inference_mode()
    def sample(
        self,
        z: Tensor,
        batch: Tensor,
        mol_embed: Tensor,
        sample_step: int = 1000,
        lattice: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample from a uniform piror distribution.

        :param z: atomic numbers;               shape: (1, n_a)
        :param batch: batch mask;               shape: (n_b, n_a, 1)
        :param mol_embed: molecule embeddings;  shape: (n_b, n_embed)
        :param sample_step: number of sampling steps
        :param lattice: lattice vectors;        shape: (n_b, 3, 3)
        :type z: torch.Tensor
        :type batch: torch.Tensor
        :type mol_embed: torch.Tensor
        :type sample_step: int
        :type lattice: torch.Tensor | None
        :return: sampled nuclear coordinates;   shape: (1, n_a, 3)
        :rtype: torch.Tensor
        """
        rho, n_a = 1, z.shape[-1]
        mu = torch.zeros((1, n_a, 3), device=self.sigma.device)
        for i in torch.linspace(1, sample_step, sample_step, device=self.sigma.device):
            t = (i - 1).view(1, 1, 1).repeat(1, n_a, 1) / sample_step
            x_hat = self.cts_output_prediction(
                z, mu, batch, mol_embed, t, 1 - self.sigma.pow(2 * t), lattice
            )
            alpha = self.sigma.pow(-2 * i / sample_step) * (
                1 - self.sigma.pow(2 / sample_step)
            )
            y = x_hat + alpha.pow(-0.5) * torch.randn_like(mu)
            mu = (rho * mu + alpha * y) / (rho + alpha)
            rho += alpha
        t_final = torch.ones((1, n_a, 1), device=self.sigma.device)
        x_hat = self.cts_output_prediction(
            z, mu, batch, mol_embed, t_final, 1 - self.sigma.pow(2 * t_final), lattice
        )
        return x_hat

    @classmethod
    def from_checkpoint(cls, ckpt: Union[str, Path]) -> Self:
        """
        Load model weight from a checkpoint.

        :param ckpt: checkpoint file
        :type ckpt: str | pathlib.Path
        :return: EGNN
        :rtype: bayesianflow_for_chem.geom.EGNN
        """
        with open(ckpt, "rb") as f:
            state = torch.load(f, "cpu", weights_only=True)
        model_nn, hparam = state["nn"], state["hparam"]
        model = cls(**hparam)
        model.load_state_dict(model_nn, True)
        return model


if __name__ == "__main__":
    ...
