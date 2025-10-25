"""
This is a compact convolutional dual-head network (CNN)

Design:
- Input: (batch, 9, 5, 5)
- Shared conv trunk -> produces feature maps (batch, filters, 5, 5)
- Policy head: 1x1 conv -> (batch, 3, 5, 5) logits -> softmax per-channel across 25 positions
- Value head: small conv(s) + linear -> scalar in (-1, 1) using tanh

Loss helper:
- policy_and_value_loss(logits, value_pred, targets, policy_coef=1.0, value_coef=1.0)
  expects targets to contain:
    - policy_targets: tensor (batch, 3, 25) with one-hot (or probability) targets per channel
    - value_targets: tensor (batch,) with -1..1 values
"""
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ConvBlock(nn.Module):
    """Basic conv -> BN -> ReLU block, keeps spatial resolution with padding=1"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class SantoNeuroNet(nn.Module):
    """
    Dual-head CNN

    Args:
        in_channels: input channels--default 9 (more in encode.py)
        filters: number of feature maps in the trunk
        n_conv_blocks: how many conv blocks in the trunk, affects receptive field
        value_hidden: hidden units in the value head before final scalar
    """

    def __init__(
        self,
        in_channels: int = 9,
        filters: int = 64,
        n_conv_blocks: int = 3,
        value_hidden: int = 128,
    ) -> None:
        super().__init__()

        # --- Shared trunk ---
        self.input_block = ConvBlock(in_channels, filters)

        trunk_blocks = []
        for _ in range(n_conv_blocks):
            trunk_blocks.append(ConvBlock(filters, filters))
        self.trunk = nn.Sequential(*trunk_blocks)

        # --- Policy head ---
        # A 1x1 conv that maps the shared filters -> 3 (one per action channel)
        # Output is logits shaped (batch, 3, 5, 5)
        self.policy_conv = nn.Conv2d(filters, 3, kernel_size=1, bias=True)

        # --- Value head ---
        # Reduce filters via 1x1 conv -> nonlinearity -> flatten -> dense -> scalar
        self.value_conv = nn.Conv2d(filters, filters // 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(filters // 2)
        self.value_act = nn.ReLU(inplace=True)
        # final linear maps flattened (filters//2 * 5 * 5) -> hidden -> scalar
        flat_size = (filters // 2) * 5 * 5
        self.value_fc1 = nn.Linear(flat_size, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        # initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialization for conv/linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, 9, 5, 5) float tensor.

        Returns:
            policy_logits: (batch, 3, 5, 5) - raw logits
            value: (batch,) - scalar value in (-1, 1)
        """
        # Shared trunk
        x = self.input_block(x)   # -> (batch, filters, 5, 5)
        x = self.trunk(x)         # -> (batch, filters, 5, 5)

        # Policy head: 1x1 conv -> (batch, 3, 5, 5)
        policy_logits = self.policy_conv(x)

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.value_act(v)  # -> (batch, filters//2, 5, 5)
        v = v.view(v.size(0), -1)  # flatten
        v = functional.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        v = torch.tanh(v).squeeze(dim=1)  # -> (batch,)
        return policy_logits, v

    # --- convenience methods for save/load/predict ---

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: Optional[int] = None) -> None:
        """
        Save model state and optionally optimizer state.
        """
        payload = {"model_state": self.state_dict()}
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if epoch is not None:
            payload["epoch"] = epoch
        torch.save(payload, path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict:
        """
        Load checkpoint and return optimizer state / epoch if present.
        """
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["model_state"])
        # Return data so caller may restore optimizer/epoch if desired
        return data

    def predict(self, x: torch.Tensor, softmax_policy: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience inference helper.

        Args:
            x: (batch, 9, 5, 5)
            softmax_policy: if True, returns per-channel probabilities, else raw logits

        Returns:
            policy: if softmax -> (batch, 3, 25) probabilities per channel; else (batch, 3, 5, 5) logits
            value: between -1 and 1
        """
        self.eval()
        with torch.no_grad():
            logits, v = self.forward(x)
            if softmax_policy:
                # collapse spatial dims and do softmax per-channel
                b = logits.shape[0]
                logits_flat = logits.view(b, 3, 25)  # (batch, 3, 25)
                # softmax along last dim for each channel independently
                probs = functional.softmax(logits_flat, dim=2)
                return probs, v
            else:
                return logits, v


def policy_and_value_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    policy_coef: float = 1.0,
    value_coef: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined policy + value loss.

    Args:
        policy_logits: (batch, 3, 5, 5) raw logits
        value_pred: (batch,) predicted scalars (already tanh-ed)
        policy_targets: (batch, 3, 25) one-hot or soft targets per channel
        value_targets: (batch,) target scalars in [-1,1]
        policy_coef/value_coef: coefficients to weight losses

    Returns:
        total_loss, info dict with components
    """
    b = policy_logits.size(0)
    logits_flat = policy_logits.view(b, 3, 25)         # (b,3,25)
    # For stability, use log_softmax and negative log likelihood with targets as probabilities:
    log_probs = functional.log_softmax(logits_flat, dim=2)     # (b,3,25)
    # policy_targets is expected as float probabilities (one-hot or smoothed)
    policy_loss_per_channel = - (policy_targets * log_probs).sum(dim=2).mean(dim=0)  # (3,)
    policy_loss = policy_loss_per_channel.sum()  # sum channels

    # value loss: MSE between prediction and target
    value_loss = functional.mse_loss(value_pred, value_targets)

    total_loss = policy_coef * policy_loss + value_coef * value_loss

    info = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "total_loss": total_loss.detach(),
    }
    return total_loss, info