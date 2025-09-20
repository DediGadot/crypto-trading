"""Simplified SAFLA neural network implementation.

The real project previously referenced an elaborate self-aware network that was
never committed.  This module provides a pragmatic PyTorch implementation that
matches the public API exercised by the tests and the rest of the codebase.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainingHistory:
    adaptation_events: list
    losses: list
    accuracies: list


class ConfidenceMetadata(dict):
    """Dictionary-like metadata that behaves as a float for comparisons."""

    def __float__(self) -> float:  # pragma: no cover - trivial conversion
        return float(self.get("confidence", 0.0))

    def _coerce(self, other: Any) -> float:
        try:
            return float(other)
        except Exception:  # pragma: no cover - defensive
            return float(self)

    def __le__(self, other: Any) -> bool:
        return float(self) <= self._coerce(other)

    def __ge__(self, other: Any) -> bool:
        return float(self) >= self._coerce(other)

    def __rle__(self, other: Any) -> bool:
        return self._coerce(other) <= float(self)

    def __rge__(self, other: Any) -> bool:
        return self._coerce(other) >= float(self)


class SAFLANeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        memory_manager: Any | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.memory_manager = memory_manager
        self._build_layers()

        # Get config for learning rate
        from ..config import get_config
        config = get_config()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.neural_learning_rate)
        self.training_history = {
            "adaptation_events": [],
            "losses": [],
            "accuracies": [],
        }
        self.performance_stats: Dict[str, Any] = {
            "batches_trained": 0,
            "last_update": None,
        }

    def _build_layers(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.layers(inputs)
        confidence = torch.sigmoid(torch.mean(outputs, dim=-1, keepdim=True))
        uncertainty = 1.0 - confidence
        meta_signal = torch.tanh(outputs)
        aux = {
            "confidence": confidence,
            "uncertainty": uncertainty,
            "meta_learning_signal": meta_signal,
        }
        return outputs, aux

    def add_hidden_layer(self, layer_size: int) -> None:
        """Add a new hidden layer to the network architecture."""
        # Rebuild layers with additional hidden layer
        original_layers = list(self.layers.children())
        layer_list = []

        # Find the output layer and insert new layer before it
        output_layer_found = False
        for i, layer in enumerate(original_layers):
            # Check if this is the final linear layer (output layer)
            if isinstance(layer, nn.Linear) and layer.out_features == self.output_dim and not output_layer_found:
                # Add new hidden layer before output layer
                layer_list.append(nn.Linear(layer.in_features, layer_size))
                layer_list.append(nn.ReLU())
                layer_list.append(nn.Dropout(self.dropout))

                # Modify output layer to accept new layer size as input
                new_output_layer = nn.Linear(layer_size, self.output_dim)
                layer_list.append(new_output_layer)
                output_layer_found = True
            else:
                layer_list.append(layer)

        # If no output layer found, append to end
        if not output_layer_found:
            layer_list.append(nn.Linear(self.hidden_dim, layer_size))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(self.dropout))
            layer_list.append(nn.Linear(layer_size, self.output_dim))

        # Update architecture
        self.layers = nn.Sequential(*layer_list)

        # Reinitialize optimizer with new parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer.param_groups[0]['lr'])

    def predict_with_confidence(
        self,
        inputs: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.eval()
        with torch.no_grad():
            outputs, aux = self.forward(inputs)
        metadata = ConfidenceMetadata(
            confidence=float(aux["confidence"].mean().item()),
            uncertainty=float(aux["uncertainty"].mean().item()),
            meta_learning_signal=float(aux["meta_learning_signal"].mean().item()),
            context_used=bool(context),
        )
        return outputs, metadata

    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.train()
        inputs = batch_data["inputs"]
        targets = batch_data["targets"]
        outputs, aux = self.forward(inputs)
        primary_loss = F.mse_loss(outputs, targets)
        confidence_term = 1.0 - torch.mean(aux["confidence"])
        uncertainty_term = torch.mean(aux["uncertainty"])
        meta_term = torch.mean(torch.abs(aux["meta_learning_signal"])) * 0.1
        total_loss = primary_loss + 0.1 * confidence_term + 0.1 * uncertainty_term + meta_term

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Accuracy proxy using cosine similarity
        with torch.no_grad():
            pred_norm = F.normalize(outputs, dim=-1)
            target_norm = F.normalize(targets, dim=-1)
            accuracy = torch.mean(torch.sum(pred_norm * target_norm, dim=-1)) * 0.5 + 0.5

        metrics = {
            "total_loss": float(total_loss.item()),
            "primary_loss": float(primary_loss.item()),
            "confidence_loss": float(confidence_term.item()),
            "uncertainty_loss": float(uncertainty_term.item()),
            "meta_loss": float(meta_term.item()),
            "accuracy": float(accuracy.item()),
            "confidence": float(aux["confidence"].mean().item()),
        }

        self.training_history["losses"].append(metrics["total_loss"])
        self.training_history["accuracies"].append(metrics["accuracy"])
        self.performance_stats["batches_trained"] += 1
        self.performance_stats["last_update"] = time.time()
        return metrics

    def adapt_to_feedback(self, feedback: Dict[str, Any]) -> bool:
        adapted = False
        if feedback.get("performance_degradation", 0) > 0.1:
            for group in self.optimizer.param_groups:
                group["lr"] *= 0.9
            adapted = True
        if feedback.get("overfitting"):
            for module in self.layers.modules():
                if isinstance(module, nn.Dropout):
                    module.p = min(0.5, module.p + 0.05)
            adapted = True
        if feedback.get('add_layer'):
            self.add_hidden_layer(feedback.get('layer_size', 64))
            adapted = True
        if adapted:
            event = {
                "timestamp": time.time(),
                "feedback": dict(feedback),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            self.training_history["adaptation_events"].append(event)
        return adapted

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "state": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.training_history,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.load_state_dict(data["state"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.training_history = data.get("history", self.training_history)

    def get_network_state(self) -> Dict[str, Any]:
        parameter_count = sum(p.numel() for p in self.parameters())
        return {
            "parameters": parameter_count,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "training_history": {
                "losses": list(self.training_history["losses"]),
                "adaptation_count": len(self.training_history["adaptation_events"]),
            },
            "performance_stats": dict(self.performance_stats),
        }


__all__ = ["SAFLANeuralNetwork"]
