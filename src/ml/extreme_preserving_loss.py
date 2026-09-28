import torch
import torch.nn as nn


class ExtremePreservingLoss(nn.Module):

    def __init__(self, extreme_threshold=50.0, extreme_weight=3.0):
        super().__init__()

        self.extreme_threshold = extreme_threshold
        self.extreme_weight = extreme_weight

        self.mse = nn.MSELoss()

    def forward(self, prediction, target):

        # Normal reconstruction error
        base_loss = self.mse(prediction, target)

        # Identify extreme-weather pixels
        extreme_mask = target >= self.extreme_threshold

        # If there are extreme pixels
        if torch.any(extreme_mask):

            extreme_prediction = prediction[extreme_mask]
            extreme_target = target[extreme_mask]

            # Give stronger penalty to errors in extreme regions
            extreme_loss = self.mse(
                extreme_prediction,
                extreme_target
            )

            total_loss = (
                base_loss
                + self.extreme_weight * extreme_loss
            )

        else:

            total_loss = base_loss

        return total_loss