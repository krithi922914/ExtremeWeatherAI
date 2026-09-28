import torch

from physics_loss import PhysicsConsistencyLoss


# ==========================================
# Create a simple weather field
# ==========================================

weather_field = torch.tensor([
    [
        [
            [20.0, 21.0, 22.0],
            [21.0, 23.0, 25.0],
            [22.0, 25.0, 28.0]
        ]
    ]
])


# ==========================================
# Create Physics Loss
# ==========================================

loss_function = PhysicsConsistencyLoss(
    physics_weight=0.1
)


# ==========================================
# Calculate Physics Loss
# ==========================================

loss = loss_function(
    weather_field
)


# ==========================================
# Display Result
# ==========================================

print("Physics Consistency Loss Test")
print("=" * 45)

print("Weather field:")
print(weather_field)

print("\nPhysics loss:")
print(loss.item())

print("\nPhysics consistency loss is working!")