# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Define ChemBFN and regressor models for training.
"""
from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional, Literal
import torch
import torch.optim as op
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import LightningModule
from .model import ChemBFN, MLP
from .scorer import Scorer

DEFAULT_MODEL_HPARAM = {"lr": 5e-5, "lr_warmup_step": 1000, "uncond_prob": 0.2}
DEFAULT_REGRESSOR_HPARAM = {
    "mode": (_mode := "regression"),
    "lr_scheduler_factor": 0.8,
    "lr_scheduler_patience": 20,
    "lr_warmup_step": 1000 if _mode == "regression" else 100,
    "max_lr": 1e-4,
    "freeze": False,
}


def _mark_only_lora_as_trainable(model: ChemBFN) -> None:
    # Modified from https://github.com/microsoft/LoRA/blob/main/loralib/utils.py
    # We only mark 'real' LoRA parameters as trainable.
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


def _lora_state_dict(model: ChemBFN) -> Dict[str, Tensor]:
    # Modified from https://github.com/microsoft/LoRA/blob/main/loralib/utils.py
    # We only loard 'real' LoRA parameters.
    state_dict = model.state_dict()
    return {k: state_dict[k] for k in state_dict if "lora_" in k}


def focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: Union[float, List[float], None] = None,
    gamma: int = 2,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    r"""
    Focal Loss implementation for binary and multi-class cases.

    .. math:: $FL = -\alpha(1 - p_t)^{gamma}ln(p_t)$

    :param inputs: predicted logits;   shape: (n_b, n_class)
    :param targets: labelled classes;  shape: (n_class)
    :param alpha: class balancing factor
    :param gamma: focusing parameter
    :param reduction: `'none'`, `'mean'` or `'sum'`; default is `'mean'`
    :type inputs: torch.Tensor
    :type targets: torch.Tensor
    :type alpha: float | list | None
    :type gamma: int
    :type reduction: str
    :return: focal loss value
    :rtype: torch.Tensor
    """
    assert inputs.dim() == 2, "The expected shape is (n_batch, n_class)."
    assert targets.dim() == 1, "The expected shape is (n_class,)."
    assert inputs.shape[0] == targets.shape[0], "Shape mismatched."
    num_class = inputs.shape[-1]
    if isinstance(alpha, (list, tuple)):
        assert num_class == (k := len(alpha)), (
            f"We have {num_class} classes but you provided {k}"
            f" balancing {'factor' if k in (0, 1) else 'factors'}."
        )
        alpha = inputs.new_tensor(alpha)
    elif isinstance(alpha, float):
        assert num_class == 2, "A float alpha is only accecpted for binary cases."
        alpha = inputs.new_tensor([1 - alpha, alpha])
    p = F.softmax(inputs, -1)
    target_onehot = F.one_hot(targets, num_class).float()
    p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
    loss = -(1 - p_t).pow(gamma) * (p_t + 1e-8).log()
    if torch.is_tensor(alpha):
        alpha_t = alpha.gather(0, targets)
        loss = alpha_t.unsqueeze(1) * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class Model(LightningModule):
    """
    Generative model class for training only.
    """

    def __init__(
        self,
        model: ChemBFN,
        mlp: Optional[MLP] = None,
        scorer: Optional[Scorer] = None,
        hparam: Optional[Dict[str, Union[int, float]]] = None,
    ) -> None:
        """
        A `~lightning.LightningModule` wrapper of bayesian flow network for chemistry model.\n
        This module is used in training stage only.
        By calling `Model(...).export_model(YOUR_WORK_DIR)` after training,
        the model(s) will be saved to `YOUR_WORK_DIR/model.pt`
        (if LoRA is enabled then `YOUR_WORK_DIR/lora.pt`) and (if exists) `YOUR_WORK_DIR/mlp.pt`.

        :param model: `~bayesianflow_for_chem.model.ChemBFN` instance.
        :param mlp: `~bayesianflow_for_chem.model.MLP` instance or `None`.
        :param scorer: `~bayesianflow_for_chem.scorer.Scorer` instance or `None`.
        :param hparam: a `dict` instance of hyperparameters.
                       See `bayesianflow_for_chem.train.DEFAULT_MODEL_HPARAM`.
        :type model: bayesianflow_for_chem.model.ChemBFN
        :type mlp: bayesianflow_for_chem.model.MLP | None
        :type scorer: bayesianflow_for_chem.scorer.Scorer | None
        :type hparam: dict | None
        """
        super().__init__()
        if hparam is None:
            hparam = DEFAULT_MODEL_HPARAM
        self.model = model
        self.mlp = mlp
        self.scorer = scorer
        self.save_hyperparameters(hparam, ignore=["model", "mlp", "scorer"])
        if model.lora_enabled:
            _mark_only_lora_as_trainable(self.model)
        self.use_scorer = self.scorer is not None

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        x = batch["token"]
        t = torch.rand((x.shape[0], 1, 1), device=x.device)
        if "mask" in batch:
            mask = batch["mask"]
        else:
            mask = None
        if self.mlp is not None and "value" in batch:
            y = batch["value"]
            y = self.mlp.forward(y)
            if y.dim() == 2:
                y = y[:, None, :]
            y_mask = F.dropout(torch.ones_like(t), self.hparams.uncond_prob, True, True)
            y_mask = (y_mask != 0).float()
            loss, p = self.model.cts_loss(x, t, y * y_mask, mask, self.use_scorer)
        else:
            loss, p = self.model.cts_loss(x, t, None, mask, self.use_scorer)
        self.log("continuous_time_loss", loss.item())
        if self.use_scorer:
            scorer_loss = self.scorer.calc_score_loss(p)
            self.log(f"{self.scorer.name}_loss", scorer_loss.item())
            loss += scorer_loss * self.scorer.eta
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
        :type workdir: pathlib.Path
        :return:
        :rtype: None
        """
        if not workdir.exists():
            workdir.mkdir()
        if self.model.lora_enabled:
            torch.save(
                {
                    "lora_nn": _lora_state_dict(self.model),
                    "lora_param": self.model.lora_param,
                },
                workdir / "lora.pt",
            )
        else:
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
    """
    Regressor class for training only.
    """

    def __init__(
        self,
        model: ChemBFN,
        mlp: MLP,
        hparam: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> None:
        """
        A `~lightning.LightningModule` wrapper of ChemBFN regression or classification model.\n
        This module is used in training stage only.
        By calling `Regressor(...).export_model(YOUR_WORK_DIR)` after training,
        the models will be saved to `YOUR_WORK_DIR/model_ft.pt`
        (or `YOUR_WORK_DIR/lora.pt` if LoRA is enabled) and `YOUR_WORK_DIR/readout.pt`.

        :param model: `~bayesianflow_for_chem.model.ChemBFN` instance.
        :param mlp: `~bayesianflow_for_chem.model.MLP` instance.
        :param hparam: a `dict` instance of hyperparameters.
                       See `bayesianflow_for_chem.train.DEFAULT_REGRESSOR_HPARAM`.
        :type model: bayesianflow_for_chem.model.ChemBFN
        :type mlp: bayesianflow_for_chem.model.MLP
        :type hparam: dict | None
        """
        super().__init__()
        if hparam is None:
            hparam = DEFAULT_REGRESSOR_HPARAM
        self.model = model
        self.mlp = mlp
        self.model.requires_grad_(not hparam["freeze"])
        self.save_hyperparameters(hparam, ignore=["model", "mlp"])
        if model.lora_enabled:
            _mark_only_lora_as_trainable(self.model)
        assert hparam["mode"] in ("regression", "classification")
        self.criteria = {"regression": F.mse_loss, "classification": F.cross_entropy}

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
        criteria = self.criteria[self.hparams.mode]
        if self.hparams.mode == "classification":
            n_b, n_y = y.shape
            z = z.reshape(n_b * n_y, -1)
            # loss = F.cross_entropy(z, y.reshape(-1).to(torch.long))
            loss = criteria(z, y.reshape(-1).to(torch.long))
        else:
            y_mask, y = self._mask_label(y)
            # loss = F.mse_loss(z * y_mask, y, reduction="mean")
            loss = criteria(z * y_mask, y, reduction="mean")
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> None:
        x, y = batch["token"], batch["value"]
        z = self.model.inference(x, self.mlp)
        if self.hparams.mode == "classification":
            n_b, n_y = y.shape
            z = z.reshape(n_b * n_y, -1)
            val_loss = 1 - (torch.argmax(z, -1) == y.reshape(-1)).float().mean()
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

        :param workdir: the directory to save the models
        :type workdir: pathlib.Path
        :return:
        :rtype: None
        """
        if not workdir.exists():
            workdir.mkdir()
        torch.save(
            {"nn": self.mlp.state_dict(), "hparam": self.mlp.hparam},
            workdir / "readout.pt",
        )
        if not self.hparams.freeze:
            if self.model.lora_enabled:
                torch.save(
                    {
                        "lora_nn": _lora_state_dict(self.model),
                        "lora_param": self.model.lora_param,
                    },
                    workdir / "lora.pt",
                )
            else:
                torch.save(
                    {"nn": self.model.state_dict(), "hparam": self.model.hparam},
                    workdir / "model_ft.pt",
                )
