import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ai_downscaling_model import WeatherDownscalingCNN


dataset_file = "data/processed/downscaling_dataset.pt"
model_file = "models/weather_downscaling_cnn.pth"


# ----------------------------------------
# Load dataset
# ----------------------------------------

data = torch.load(
    dataset_file,
    map_location="cpu"
)

X = data["inputs"]
Y = data["targets"]


print("AI Weather Downscaling Training")
print("=" * 45)

print()
print("Input shape :", X.shape)
print("Target shape:", Y.shape)


# ----------------------------------------
# Create DataLoader
# ----------------------------------------

dataset = TensorDataset(X, Y)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# ----------------------------------------
# Create model
# ----------------------------------------

model = WeatherDownscalingCNN()


# ----------------------------------------
# Training configuration
# ----------------------------------------

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 100


# ----------------------------------------
# Train model
# ----------------------------------------

print()
print("Starting training...")
print()


for epoch in range(epochs):

    total_loss = 0.0

    for inputs, targets in loader:

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = loss_function(
            predictions,
            targets
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_loss = (
        total_loss / len(loader)
    )


    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {average_loss:.6f}"
        )


# ----------------------------------------
# Save trained model
# ----------------------------------------

torch.save(
    model.state_dict(),
    model_file
)


print()
print("Training completed!")
print()
print("Model saved successfully!")
print("File:", model_file)