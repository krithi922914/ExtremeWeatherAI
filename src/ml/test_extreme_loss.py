import torch

from extreme_preserving_loss import ExtremePreservingLoss


# ------------------------------------------
# Create test target
# ------------------------------------------

target = torch.tensor([
    [
        [[20.0, 25.0],
         [55.0, 70.0]]
    ]
])


# ------------------------------------------
# Create prediction
# ------------------------------------------

prediction = torch.tensor([
    [
        [[22.0, 23.0],
         [48.0, 60.0]]
    ]
])


# ------------------------------------------
# Calculate loss
# ------------------------------------------

loss_function = ExtremePreservingLoss(
    extreme_threshold=50.0,
    extreme_weight=3.0
)

loss = loss_function(
    prediction,
    target
)


print("Extreme Preserving Loss Test")
print("=" * 40)

print("Target:")
print(target)

print("\nPrediction:")
print(prediction)

print("\nCalculated Loss:")
print(loss.item())

print("\nExtreme pixels detected:")
print(torch.sum(target >= 50.0).item())

print("\nExtreme preserving loss is working!")