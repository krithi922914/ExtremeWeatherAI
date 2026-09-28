import torch
import torch.nn as nn

from extreme_preserving_loss import ExtremePreservingLoss
from physics_loss import PhysicsConsistencyLoss


class CombinedWeatherLoss(nn.Module):

    def __init__(
        self,
        extreme_threshold=50.0,
        extreme_weight=3.0,
        physics_weight=0.1
    ):
        super().__init__()

        self.extreme_loss = ExtremePreservingLoss(
            extreme_threshold=extreme_threshold,
            extreme_weight=extreme_weight
        )

        self.physics_loss = PhysicsConsistencyLoss(
            physics_weight=physics_weight
        )

    def forward(self, prediction, target):

        # Extreme-aware reconstruction loss
        reconstruction_loss = self.extreme_loss(
            prediction,
            target
        )

        # Physics consistency loss
        physics_loss = self.physics_loss(
            prediction
        )

        # Total loss
        total_loss = (
            reconstruction_loss
            + physics_loss
        )

        return total_loss