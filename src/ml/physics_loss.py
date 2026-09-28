import torch
import torch.nn as nn


class PhysicsConsistencyLoss(nn.Module):

    def __init__(self, physics_weight=0.1):
        super().__init__()

        self.physics_weight = physics_weight

    def spatial_gradient_loss(self, field):

        # Gradient along latitude direction
        gradient_y = field[:, :, 1:, :] - field[:, :, :-1, :]

        # Gradient along longitude direction
        gradient_x = field[:, :, :, 1:] - field[:, :, :, :-1]

        # Penalize excessively large spatial changes
        loss_x = torch.mean(gradient_x ** 2)

        loss_y = torch.mean(gradient_y ** 2)

        return loss_x + loss_y

    def forward(self, prediction):

        physics_loss = self.spatial_gradient_loss(
            prediction
        )

        return self.physics_weight * physics_loss