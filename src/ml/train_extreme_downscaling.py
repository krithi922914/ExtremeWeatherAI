import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_downscaling_model import WeatherDownscalingCNN
from extreme_preserving_loss import ExtremePreservingLoss


# ==========================================
# Load Dataset
# ==========================================

data = torch.load(
    "data/processed/downscaling_dataset.pt",
    map_location="cpu"
)

X = data["inputs"]
Y = data["targets"]

print("Extreme-Preserving AI Downscaling Training")
print("=" * 50)

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
# Extreme-Preserving Loss
# ==========================================

loss_function = ExtremePreservingLoss(
    extreme_threshold=50.0,
    extreme_weight=3.0
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

        # Calculate extreme-aware loss
        loss = loss_function(
            predictions,
            targets
        )

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/100 | "
            f"Extreme Loss: {average_loss:.6f}"
        )


# ==========================================
# Save Model
# ==========================================

torch.save(
    model.state_dict(),
    "models/extreme_weather_downscaling_cnn.pth"
)


print("\nTraining completed!")

print(
    "\nExtreme-aware model saved successfully!"
)

print(
    "File: models/extreme_weather_downscaling_cnn.pth"
)