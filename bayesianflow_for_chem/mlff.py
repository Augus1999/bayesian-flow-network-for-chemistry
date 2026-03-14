# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa Sueno)
"""
Machine learning force field modules.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Callable, Self
import torch
from torch import nn, Tensor
from torch.autograd import grad
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


_ORBITALS = "1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p 6f 7d 7f".split()
_POSSIBLE_ELECTRONS = {"s": 2, "p": 6, "d": 10, "f": 14}


def _electron_config(atomic_num: int) -> List[int]:
    """
    Generate electron configuration for a given atomic number.

    :param atomic_num: atomic number
    :type atomic_num: int
    :return: electron configuration
    :rtype: list
    """
    config = []
    electron_count, last_idx = 0, -1
    for i in _ORBITALS:
        if electron_count < atomic_num:
            config.append(_POSSIBLE_ELECTRONS[i[-1]])
            electron_count += _POSSIBLE_ELECTRONS[i[-1]]
            last_idx += 1
        else:
            config.append(0)
    if electron_count > atomic_num:
        config[last_idx] -= electron_count - atomic_num
    return config


def loss_calc(
    out: Dict[str, Tensor],
    label: Dict[str, Tensor],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> Dict[str, Tensor]:
    """
    Calculate the scalar/vector loss based on key(s) in label.

    :param out: output of model
    :param label: training set label
    :param loss_fn: loss function
    :type out: dict
    :type label: dict
    :loss_fn: callable
    :return: calculated loss
    :rtype: dict
    """
    loss: Dict[str, Tensor] = {}
    if "scalar" in label:
        loss["scalar"] = loss_fn(out["scalar"], label["scalar"])
    if "vector" in label:
        loss["vector"] = loss_fn(out["vector"], label["vector"])
    return loss


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
        :return: cutoff mask;  shape: (1, n_a, k, 1)
        :rtype: torch.Tensor
        """
        cutoff = 0.5 * (torch.pi * d / self.cutoff).cos() + 0.5
        return cutoff.masked_fill(d > self.cutoff, 0)[..., None]


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
                 edge vectors;  shape: (1, n_a, k, 3, 1) \n
                 edge indices;  shape: (1, n_a, k)
        :rtype: tuple
        """
        n_a = r.shape[1]
        k = min(self.k, n_a - 1)
        vec = r[:, :, None, :] - r[:, None, :, :]
        vec.masked_fill(batch_mask, torch.inf)  # mask the 'off-diagonal' elements
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
            ds = torch.norm(vecs, 2, -1)
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
            d = torch.norm(vec, 2, -1)
        edge = (-d).topk(k, dim=-1)
        d, idxs = -edge.values, edge.indices
        vec_idxs = idxs[..., None].repeat(1, 1, 1, 3)
        vec = vec.masked_fill(vec == torch.inf, 0).gather(dim=-2, index=vec_idxs)
        vec = (vec / d.masked_fill(d == 0, torch.inf)[..., None])[..., None]
        return d, vec, idxs


class Message(nn.Module):
    """
    Message block for PAINN.
    """

    def __init__(
        self, cutoff_radius: float = 5.0, num_kernel: int = 64, channel: int = 128
    ) -> None:
        """
        Message block.

        :param cutoff_radius: cutoff radius
        :param num_kernel: number of RBF features
        :param channel: hidden layer features
        :type cutoff_radius: float
        :type num_kernel: int
        :type channel: int
        """
        super().__init__()
        self.rbf = RBF(cutoff_radius, num_kernel)
        self.proj_rbf = nn.Linear(num_kernel, channel * 3)
        self.proj_s = nn.Sequential(
            nn.Linear(channel, channel), nn.SiLU(), nn.Linear(channel, channel * 3)
        )

    def forward(
        self, s: Tensor, v: Tensor, d: Tensor, vec: Tensor, idx: Tensor, cutoff: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        :param s: scalar message;     shape: (1, n_a, n_f)
        :param v: vector message;     shape: (1, n_a, 3, n_f)
        :param d: pairwise distance;  shape: (1, n_a, k)
        :param vec: edge vectors;     shape: (1, n_a, k, 3, 1)
        :param idx: edge indices;     shape: (1, n_a, k)
        :param cutoff: cutoff mask;   shape: (1, n_a, k, 1)
        :type s: torch.Tensor
        :type v: torch.Tensor
        :type d: torch.Tensor
        :type vec: torch.Tensor
        :type idx: torch.Tensor
        :type cutoff: torch.Tensor
        :return: updated scalar message;  shape: (1, n_a, n_f) \n
                 updated vector message;  shape: (1, n_a, 3, n_f)
        :rtype: tuple
        """
        w = self.proj_rbf(self.rbf(d)) * cutoff
        phi = self.proj_s(s)[0][idx]
        m_s, m_vs, m_vv = (phi * w).chunk(3, -1)  # shape: (1, n_a, k, n_f)
        ds = m_s.sum(2)
        dv = (m_vs.unsqueeze(3) * vec + m_vv.unsqueeze(3) * v[0][idx]).sum(2)
        return s + ds, v + dv


class Update(nn.Module):
    """
    Update block of PAINN.
    """

    def __init__(self, channel: int = 128) -> None:
        """
        Update block.

        :param channel: hidden layer features
        :type channel: int
        """
        super().__init__()
        self.proj_uv = nn.Linear(channel, channel * 2)
        self.proj_a = nn.Sequential(
            nn.Linear(channel * 2, channel), nn.SiLU(), nn.Linear(channel, channel * 3)
        )

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param s: scalar message;         shape: (1, n_a, n_f)
        :param v: vector message;         shape: (1, n_a, 3, n_f)
        :type s: torch.Tensor
        :type v: torch.Tensor
        :return: updated scalar message;  shape: (1, n_a, n_f) \n
                 updated vector message;  shape: (1, n_a, 3, n_f)
        :rtype: tuple
        """
        u_v, v_v = self.proj_uv(v).chunk(2, -1)
        a_vv, a_sv, a_ss = self.proj_a(
            torch.cat([s, (v_v.pow(2).sum(2) + 1e-8).sqrt()], -1)
        ).chunk(3, -1)
        ds = (u_v * v_v).sum(2) * a_sv + a_ss
        dv = a_vv.unsqueeze(2) * u_v
        return s + ds, v + dv


class PAINN(nn.Module):
    """
    PAINN module.
    """

    def __init__(
        self,
        num_embed: int = 120,
        channel: int = 128,
        cutoff_radius: float = 5.0,
        num_kernel: int = 64,
        max_neighbour: int = 15,
        num_layer: int = 3,
    ) -> None:
        """
        PAINN representation.

        :param num_embed: number of embedded elements
        :param channel: hidden layer features
        :param cutoff_radius: piarwise distance cutoff radius
        :param num_kernel: number of RBF features
        :param max_neighbour: maximum number of atom-nighbours
        :param num_layer: number of Interaction blocks
        :type num_embed: int
        :type channel: int
        :type cutoff_radius: float
        :type num_kernel: int
        :type max_neighbour: int
        :type num_layer: int
        """
        super().__init__()
        self.num_layer = num_layer
        self.embed_a = nn.Embedding(num_embed, channel, 0)
        self.embed_b = nn.Sequential(
            nn.Embedding.from_pretrained(
                torch.tensor(
                    [_electron_config(i) for i in range(num_embed)], dtype=torch.float32
                )
            ),
            nn.Linear(22, channel),
        )
        self.distance = Distance(max_neighbour)
        self.cutoff = CosinCutOff(cutoff_radius)
        self.message_layers = nn.ModuleList(
            [Message(cutoff_radius, num_kernel, channel) for _ in range(num_layer)]
        )
        self.update_layers = nn.ModuleList([Update(channel) for _ in range(num_layer)])
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel), nn.SiLU(), nn.Linear(channel, 1)
        )
        self.hparam = {
            "num_embed": num_embed,
            "channel": channel,
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
        lattice: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        :param z: atomic numbers;               shape: (1, n_a)
        :param r: nuclear coordinates;          shape: (1, n_a, 3)
        :param batch: batch mask;               shape: (n_b, n_a, 1)
        :param lattice: lattice vectors;        shape: (n_b, 3, 3)
        :type z: torch.Tensor
        :type r: torch.Tensor
        :type batch: torch.Tensor
        :type lattice: torch.Tensor | None
        :return: energies;       shape: (n_b, 1) \n
                 atomic forces;  shape: (1, n_a, 3)
        :rtype: tuple
        """
        r.requires_grad_(True)
        if lattice is not None:
            lattice = (lattice[:, None, :, :] * batch[..., None]).sum(0, True)
        _b = batch.squeeze(-1)
        batch_mask = (_b.transpose(-2, -1) @ _b == 0)[None, :, :, None]
        d, vec, idx = self.distance(r, batch_mask, lattice)
        cutoff = self.cutoff(d)
        s = self.embed_a(z) + self.embed_b(z)
        v = torch.zeros_like(r)[..., None].repeat(1, 1, 1, s.shape[-1])
        for i in range(self.num_layer):
            s, v = self.message_layers[i](s, v, d, vec, idx, cutoff)
            s, v = self.update_layers[i](s, v)
        y = self.mlp(s)
        y = (y.repeat(batch.shape[0], 1, 1) * batch).sum(dim=-2)
        grad_outputs: List[Optional[Tensor]] = [torch.ones_like(y)]
        dy = grad(
            outputs=[y],
            inputs=[r],
            grad_outputs=grad_outputs,
            retain_graph=self.training,
            create_graph=self.training,
        )[0]
        return y, -dy

    @classmethod
    def from_checkpoint(cls, ckpt: Union[str, Path]) -> Self:
        """
        Load model weight from a checkpoint.

        :param ckpt: checkpoint file
        :type ckpt: str | pathlib.Path
        :return: PAINN
        :rtype: bayesianflow_for_chem.mlff.PAINN
        """
        with open(ckpt, "rb") as f:
            state = torch.load(f, "cpu", weights_only=True)
        model_nn, hparam = state["nn"], state["hparam"]
        model = cls(**hparam)
        model.load_state_dict(model_nn, True)
        return model


class MLFF(Calculator):
    """
    Machine Learning Force Field implementation.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_file_name: Union[str, Path],
        scale: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        ASE calculator class wrapper of PAINN model.

        :param model_file_name: a path to trained model <file>
        :param scale: energy scaling factor
        :param device: hardware accelerator
        :type model_file_name: str | pathlib.Path
        :type scale: float
        :type device: str | torch.device
        """
        super().__init__()
        self.model = PAINN.from_checkpoint(model_file_name).to(device).eval()
        self.scale = scale
        self.device = device

    def calculate(
        self,
        atoms: Atoms,
        properties: Optional[List[str]] = None,
        system_changes: Optional[List[str]] = None,
    ) -> None:
        """
        Calculate the properties.

        :param atoms: an `~ase.Atoms` instance
        :param properties: implemented properties (not in use)
        :param system_changes: list of changes for ASE (not in use)
        :type atoms: ase.Atoms
        :type properties: list | None
        :type system_changes: list | None
        :return:
        :rtype: None
        """
        if properties is None:
            properties = ["energy", "forces"]
        if system_changes is None:
            system_changes = all_changes
        atoms_ = atoms.copy()
        z = torch.tensor(atoms_.numbers, dtype=torch.long, device=self.device)[None, :]
        r = torch.tensor(atoms_.positions, dtype=torch.float32, device=self.device)[
            None, ...
        ]
        batch = torch.ones_like(z, dtype=torch.float32).unsqueeze(dim=-1)
        lattice = torch.tensor(atoms_.cell.array, dtype=torch.float32)
        if lattice.abs().sum() > 0:
            pbc = torch.tensor(atoms_.pbc, dtype=torch.float32)
            lattice = (lattice * pbc[:, None])[None, ...].to(self.device)
        else:
            lattice = None
        s, v = self.model.forward(z, r, batch, lattice)
        results = {
            "energy": s.detach().cpu().item() * self.scale,
            "forces": v[0].numpy(force=True) * self.scale,
        }
        self.results = results


if __name__ == "__main__":
    ...
