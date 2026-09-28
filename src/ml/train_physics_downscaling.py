import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_downscaling_model import WeatherDownscalingCNN
from combined_weather_loss import CombinedWeatherLoss


# ==========================================
# Load Dataset
# ==========================================

data = torch.load(
    "data/processed/downscaling_dataset.pt",
    map_location="cpu"
)

X = data["inputs"]
Y = data["targets"]

print("Physics + Extreme-Aware Downscaling Training")
print("=" * 55)

print("Input shape :", X.shape)
print("Target shape:", Y.shape)


# ==========================================
# Create DataLoader
# ==========================================

dataset = TensorDataset(X, Y)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# ==========================================
# Create Model
# ==========================================

model = WeatherDownscalingCNN()


# ==========================================
# Combined Loss
# ==========================================

loss_function = CombinedWeatherLoss(
    extreme_threshold=50.0,
    extreme_weight=3.0,
    physics_weight=0.1
)


# ==========================================
# Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# Training
# ==========================================

epochs = 100

print("\nStarting training...\n")


for epoch in range(epochs):

    total_loss = 0.0

    for inputs, targets in loader:

        # Forward pass
        predictions = model(inputs)

        # Combined reconstruction + extreme + physics loss
        loss = loss_function(
            predictions,
            targets
        )

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/100 | "
            f"Combined Loss: {average_loss:.6f}"
        )


# ==========================================
# Save Model
# ==========================================

torch.save(
    model.state_dict(),
    "models/physics_extreme_downscaling_cnn.pth"
)


print("\nTraining completed!")

print(
    "\nPhysics + extreme-aware model saved successfully!"
)

print(
    "File: models/physics_extreme_downscaling_cnn.pth"
)