import torch

from combined_weather_loss import CombinedWeatherLoss


# ==========================================
# Create Test Prediction and Target
# ==========================================

target = torch.tensor([
    [
        [
            [20.0, 25.0, 30.0],
            [25.0, 55.0, 65.0],
            [30.0, 60.0, 70.0]
        ]
    ]
])


prediction = torch.tensor([
    [
        [
            [21.0, 24.0, 28.0],
            [24.0, 50.0, 60.0],
            [31.0, 56.0, 64.0]
        ]
    ]
])


# ==========================================
# Create Combined Loss
# ==========================================

loss_function = CombinedWeatherLoss(
    extreme_threshold=50.0,
    extreme_weight=3.0,
    physics_weight=0.1
)


# ==========================================
# Calculate Loss
# ==========================================

loss = loss_function(
    prediction,
    target
)


# ==========================================
# Display Result
# ==========================================

print("Combined Weather Loss Test")
print("=" * 50)

print("Target shape:")
print(target.shape)

print("\nPrediction shape:")
print(prediction.shape)

print("\nCombined loss:")
print(loss.item())

print("\nCombined loss is working!")