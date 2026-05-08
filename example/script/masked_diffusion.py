# -*- coding: utf-8 -*-
# Author: Nianze TAO
"""
This is the minimum example of modifing ChemBFN to be compatible with LLaDA training objective.
"""

from typing import Tuple, Optional, Dict
import torch
from torch import nn, Tensor
from bayesianflow_for_chem import ChemBFN
from bayesianflow_for_chem.data import VOCAB_COUNT, VOCAB_KEYS
from bayesianflow_for_chem.tool import sample
from bayesianflow_for_chem.train import Model


def _get_num_transfer_tokens(mask_index: Tensor, steps: int) -> Tensor:
    """
    See: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
    """
    mask_num = mask_index.sum(1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


class LLaDA(ChemBFN):
    def __init__(self, num_vocab: int, **kargs) -> None:
        super().__init__(num_vocab + 1, **kargs)
        self.embedding = nn.Embedding(num_vocab + 1, self.embedding.weight.shape[0], 0)
        self.hparam["num_vocab"] = num_vocab

    def _forward(
        self, x: Tensor, t: Tensor, eps: float = 1e-3
    ) -> Tuple[Tensor, Tensor, Tensor]:
        b, l = x.shape
        p_mask = (1 - eps) * t.squeeze() + eps
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=x.device) < p_mask
        # K - 1 is used for <mask> token
        noisy_batch = torch.where(masked_indices, self.K - 1, x)
        return noisy_batch, masked_indices, p_mask

    def calc_loss(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        noisy_batch, masked_indices, p_mask = self._forward(x, t)
        logits = self.forward(noisy_batch, t, None, y)
        token_loss = (
            torch.nn.functional.cross_entropy(
                logits[masked_indices], x[masked_indices], reduction="none"
            )
            / p_mask[masked_indices]
        )
        loss = token_loss.sum() / (x.shape[0] * x.shape[1])
        return loss

    def sample(
        self,
        batch_size: int,
        sequence_size: int,
        y: Optional[Tensor],
        sample_step: int = 100,
        guidance_strength: float = 4.0,
        token_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x = torch.ones(
            (batch_size, sequence_size), dtype=torch.long, device=self.beta.device
        )
        x = (self.K - 1) * x
        num_transfer_tokens = _get_num_transfer_tokens(x == (self.K - 1), sample_step)
        for i, t in enumerate(
            torch.linspace(0, 1, sample_step, device=self.beta.device)
        ):
            t = t.view(1, 1, 1).repeat(batch_size, 1, 1)
            mask_index = x == (self.K - 1)
            logits = self.forward(x, t, None, None)
            if y is not None:
                cond_logits = self.forward(x, t, None, y)
                logits = logits + (guidance_strength + 1) * (cond_logits - logits)
            if token_mask is not None:
                logits = logits.masked_fill_(token_mask, 0.0)
            x0 = torch.argmax(logits, dim=-1)  # b, l
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            x0_p[:, sequence_size:] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
        return x, None


class LLaDAModel(Model):
    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        x = batch["token"]
        t = torch.rand((x.shape[0], 1, 1), device=x.device)
        if self.mlp is not None and "value" in batch:
            y = batch["value"]
            y = self.mlp.forward(y)
            if y.dim() == 2:
                y = y[:, None, :]
            y_mask = torch.nn.functional.dropout(
                torch.ones_like(t), self.hparams.uncond_prob, True, True
            )
            y_mask = (y_mask != 0).float()
            loss = self.model.calc_loss(x, t, y * y_mask)
        else:
            loss = self.model.calc_loss(x, t, None)
        self.log("loss", loss.item())
        return loss


lmax = 111

if __name__ == "__main__":
    model = LLaDA(VOCAB_COUNT)
    # or model = LLaDA.from_checkpoint("YOUR/MODEL/DIR.pt")
    print(sample(model, 2, lmax, vocab_keys=VOCAB_KEYS + ["<mask>"]))
