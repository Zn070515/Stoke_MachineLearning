"""PyTorch Lightning module wrapping stock prediction training loop.

Model-agnostic: accepts any nn.Module via the `model` argument.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from stoke_ml.evaluation.metrics import mcc_score


class StockLightningModule(pl.LightningModule):
    """Lightning wrapper for stock prediction models (LSTM, Transformer, etc.)."""

    def __init__(
        self,
        model: nn.Module | None = None,
        input_dim: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weight: list[float] | None = None,
        use_scheduler: bool = True,
        model_type: str = "lstm",
    ):
        super().__init__()
        self.save_hyperparameters()

        if model is not None:
            self.model = model
        elif model_type == "transformer":
            from stoke_ml.models.dl.transformer_model import TransformerModel
            self.model = TransformerModel(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=8,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_type == "attention":
            from stoke_ml.models.dl.attention_model import SimpleAttentionModel
            self.model = SimpleAttentionModel(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=4,
                dropout=dropout,
            )
        else:
            from stoke_ml.models.dl.lstm_model import LSTMModel
            self.model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        if class_weight is not None:
            self._class_weight = torch.tensor(class_weight, dtype=torch.float)
        else:
            self._class_weight = None
        self._use_scheduler = use_scheduler
        self._criterion = nn.CrossEntropyLoss(weight=self._class_weight)
        self._val_preds = []
        self._val_targets = []

    def _ensure_criterion_device(self, device):
        if self._class_weight is not None and self._class_weight.device != device:
            self._class_weight = self._class_weight.to(device)
            self._criterion = nn.CrossEntropyLoss(weight=self._class_weight)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self._ensure_criterion_device(x.device)
        logits = self(x)
        loss = self._criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self._ensure_criterion_device(x.device)
        logits = self(x)
        loss = self._criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)
        self._val_preds.append(preds.cpu().numpy())
        self._val_targets.append(y.cpu().numpy())
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            all_preds = np.concatenate(self._val_preds)
            all_targets = np.concatenate(self._val_targets)
            mcc = mcc_score(all_targets, all_preds)
            self.log("val_mcc", mcc, on_epoch=True)
            self._val_preds.clear()
            self._val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if not self._use_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
