"""
This is a CNN for Santorini, using value-only branch for move eval

Design:
- Input: (batch, 14, 5, 5) - board state (11 channels) + action encoding (3 channels)
- Shared conv trunk -> produces feature maps (batch, filters, 5, 5)
- Value head: conv reduction + linear layers -> scalar in (-1, 1) using tanh
"""
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ConvBlock(nn.Module):
    """Basic conv -> BN -> ReLU block, keeps cnn from making the board smaller"""

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
    CNN

    Args:
        in_channels: input channels - 14 (more in encode.py)
        filters: number of feature maps in the trunk
        n_conv_blocks: how many conv blocks in the trunk, affects receptive field
        value_hidden: hidden units in the value head before final scalar
    """

    def __init__(
        self,
        in_channels: int = 14,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 14, 5, 5) float tensor containing:
               - Channels 0-10: board state (heights 1-3, dome, workers of each player, active player)
               - Channels 11-14: action encoding (worker pos, move pos, build pos)

        Returns:
            value: (batch,) - scalar value in (-1, 1)
        """
        # Shared trunk
        x = self.input_block(x)   # -> (batch, filters, 5, 5)
        x = self.trunk(x)         # -> (batch, filters, 5, 5)

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.value_act(v)  # -> (batch, filters//2, 5, 5)
        v = v.view(v.size(0), -1)  # flatten
        v = functional.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        v = torch.tanh(v).squeeze(dim=1)  # -> (batch,)
        return v

    # convenience methods for save/load/predict

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: Optional[int] = None,**kwargs) -> None:
        """
        Save model state and optionally optimizer state.
        """
        payload = {"model_state": self.state_dict()}
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if epoch is not None:
            payload["epoch"] = epoch
        payload.update(kwargs)  # extra data
        torch.save(payload, path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict:
        """
        Load checkpoint and return optimizer state / epoch if present.
        """
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["model_state"])
        return data

    @torch.no_grad() #disables gradient calculation
    def evaluate_actions(self, board_state: torch.Tensor, action_encodings:torch.Tensor) -> torch.Tensor:
        """
                Evaluate a batch of actions for the same board state.

                Args:
                    board_state: (11, 5, 5) single board state
                    action_encodings: (num_actions, 3, 5, 5) action encodings

                Returns:
                    values: (num_actions,) scores for each action
                """
        self.eval()

        # Expand board state to match number of actions
        num_actions = action_encodings.size(0)
        board_batch = board_state.unsqueeze(0).expand(num_actions, -1, -1, -1)

        # Concatenate board + actions
        inputs = torch.cat([board_batch, action_encodings], dim=1)  # (num_actions, 14, 5, 5)

        # Evaluate
        values = self.forward(inputs)
        return values


def value_loss(
    value_pred: torch.Tensor,
    value_targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute value loss

    Args:
        value_pred: (batch,) predicted scalars (already tanh in [-1,1])
        value_targets: (batch,) target scalars in [-1,1]
        Supposed to be positive for good moves, negative for bad ones, 0 for meh

    Returns:
        Dict with loss and mean absolute error
    """
    loss = functional.mse_loss(value_pred, value_targets)
    mae = functional.l1_loss(value_pred, value_targets)
    return {"loss": loss, "mae": mae.detach()}

# Empty model creation
if __name__ == "__main__":
    # Create model
    model = SantoNeuroNet(
        in_channels=14,
        filters=64,
        n_conv_blocks=3,
        value_hidden=128
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Example forward pass
    batch_size = 1
    x = torch.randn(batch_size, 14, 5, 5)
    values = model(x)
    model.save_checkpoint("ml/learned_models/best.pt")

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {values.shape}")
    print(f"Output range: [{values.min():.3f}, {values.max():.3f}]")