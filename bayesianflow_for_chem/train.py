# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Define ChemBFN and regressor models for training.
"""
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import torch
import torch.optim as op
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import LightningModule
from .model import ChemBFN, MLP

DEFAULT_MODEL_HPARAM = {"lr": 5e-5, "lr_warmup_step": 1000, "uncond_prob": 0.1}
DEFAULT_REGRESSOR_HPARAM = {
    "mode": "regression",
    "lr_scheduler_factor": 0.8,
    "lr_scheduler_patience": 20,
    "lr_warmup_step": 1000,
    "max_lr": 1e-4,
    "freeze": False,
}


class Model(LightningModule):
    def __init__(
        self,
        model: ChemBFN,
        mlp: Optional[MLP] = None,
        hparam: Dict[str, Union[int, float]] = DEFAULT_MODEL_HPARAM,
    ) -> None:
        """
        A `~lightning.LightningModule` wrapper of bayesian flow network for chemistry model.\n
        This module is used in training stage only. By calling `Model(...).export_model(YOUR_WORK_DIR)` after training,
        the model(s) will be saved to `YOUR_WORK_DIR/model.pt` and (if exists) `TOUR_WORK_DIR/cnet.pt`.

        :param model: `~bayesianflow_for_chem.model.ChemBFN` instance.
        :param mlp: `~bayesianflow_for_chem.model.MLP` instance or `None`.
        :param hparam: a `dict` instance of hyperparameters. See `bayesianflow_for_chem.train.DEFAULT_MODEL_HPARAM`.
        """
        super().__init__()
        self.model = model
        self.mlp = mlp
        self.save_hyperparameters(hparam, ignore=["model", "mlp"])

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        x = batch["token"]
        t = torch.rand((x.shape[0], 1), device=x.device)
        if self.mlp is not None:
            y = batch["value"]
            y = self.mlp.forward(y)
            if y.dim() == 2:
                y = y[:, None, :]
            y_mask = F.dropout(torch.ones_like(t), self.hparams.uncond_prob, True, True)
            y_mask = (y_mask != 0).float()[..., None]
            loss = self.model.cts_loss(x, t, y * y_mask)
        else:
            loss = self.model.cts_loss(x, t, None)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self) -> Dict[str, op.AdamW]:
        optimizer = op.AdamW(self.parameters(), lr=1e-8, weight_decay=0.01)
        return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs) -> None:
        optimizer: op.AdamW = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        # warm-up step
        if self.trainer.global_step < self.hparams.lr_warmup_step:
            lr_scale = int(self.trainer.global_step + 1) / self.hparams.lr_warmup_step
            lr_scale = min(1.0, lr_scale)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad(set_to_none=True)

    def export_model(self, workdir: Path) -> None:
        """
        Save the trained model.

        :param workdir: the directory to save the model(s)
        :return: None
        """
        torch.save(
            {"nn": self.model.state_dict(), "hparam": self.model.hparam},
            workdir / "model.pt",
        )
        if self.mlp is not None:
            torch.save(
                {"nn": self.mlp.state_dict(), "hparam": self.mlp.hparam},
                workdir / "mlp.pt",
            )


class Regressor(LightningModule):
    def __init__(
        self,
        model: ChemBFN,
        mlp: MLP,
        hparam: Dict[str, Union[str, int, float, bool]] = DEFAULT_REGRESSOR_HPARAM,
    ) -> None:
        """
        A `~lightning.LightningModule` wrapper of bayesian flow network for chemistry regression model.\n
        This module is used in training stage only. By calling `Regressor(...).export_model(YOUR_WORK_DIR)` after training,
        the models will be saved to `YOUR_WORK_DIR/model.pt` and `TOUR_WORK_DIR/readout.pt`.

        :param model: `~bayesianflow_for_chem.model.ChemBFN` instance.
        :param cnet: `~bayesianflow_for_chem.model.MLP` instance or `None`.
        :param hparam: a `dict` instance of hyperparameters. See `bayesianflow_for_chem.train.DEFAULT_REGRESSOR_HPARAM`.
        """
        super().__init__()
        self.model = model
        self.mlp = mlp
        self.model.requires_grad_(not hparam["freeze"])
        self.save_hyperparameters(hparam, ignore=["model", "mlp"])

    @staticmethod
    def _mask_label(label: Tensor) -> Tuple[Tensor, Tensor]:
        # find the unlabelled position(s)
        label_mask = (label != torch.inf).float()
        # masked the unlabelled position(s)
        masked_label = label.masked_fill(label == torch.inf, 0)
        return label_mask, masked_label

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        x, y = batch["token"], batch["value"]
        z = self.model.inference(x, self.mlp)
        if self.hparams.mode == "classification":
            loss = F.cross_entropy(z, y.reshape(-1).to(torch.long))
        else:
            y_mask, y = self._mask_label(y)
            loss = F.mse_loss(z * y_mask, y, reduction="mean")
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> None:
        x, y = batch["token"], batch["value"]
        z = self.model.inference(x, self.mlp)
        if self.hparams.mode == "classification":
            val_loss = 1 - (torch.argmax(z, -1) == y).float().mean()
        else:
            y_mask, y = self._mask_label(y)
            val_loss = (z * y_mask - y).abs().sum() / y_mask.sum()
        self.log("val_loss", val_loss.item())

    def configure_optimizers(self) -> Dict:
        optimizer = op.AdamW(self.parameters(), lr=1e-7, weight_decay=0.01)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_scheduler_factor,
                patience=self.hparams.lr_scheduler_patience,
                min_lr=1e-6,
            ),
            "interval": "epoch",
            "monitor": "val_loss",
            "frequency": 1,
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def optimizer_step(self, *args, **kwargs) -> None:
        optimizer: op.AdamW = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        # warm-up step
        if self.trainer.global_step < self.hparams.lr_warmup_step:
            lr_scale = int(self.trainer.global_step + 1) / self.hparams.lr_warmup_step
            lr_scale = min(1.0, lr_scale)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.max_lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad(set_to_none=True)

    def export_model(self, workdir: Path) -> None:
        """
        Save the trained model.

        :param workdir: the directory to save the model
        :return: None
        """
        torch.save(
            {"nn": self.mlp.state_dict(), "hparam": self.mlp.hparam},
            workdir / "readout.pt",
        )
        if not self.hparams.freeze:
            torch.save(
                {"nn": self.model.state_dict(), "hparam": self.model.hparam},
                workdir / "model_ft.pt",
            )
