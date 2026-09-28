import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gnn_model import WeatherGNN


input_file = "data/processed/trajectory_dataset.csv"


# ----------------------------------------
# Load trajectory data
# ----------------------------------------

records = []

with open(input_file, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        records.append({
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "wind": float(row["max_wind"]),
            "intensity": float(row["intensity_score"])
        })


# ----------------------------------------
# Create training samples
# ----------------------------------------

training_samples = []

for i in range(len(records) - 2):

    previous = records[i]
    current = records[i + 1]
    target = records[i + 2]

    features = np.array([
        [
            previous["latitude"],
            previous["longitude"],
            previous["wind"],
            previous["intensity"]
        ],
        [
            current["latitude"],
            current["longitude"],
            current["wind"],
            current["intensity"]
        ]
    ], dtype=np.float32)

    target_location = np.array([
        target["latitude"],
        target["longitude"]
    ], dtype=np.float32)

    training_samples.append(
        (features, target_location)
    )


# ----------------------------------------
# Convert to PyTorch tensors
# ----------------------------------------

X = torch.tensor(
    np.array([sample[0] for sample in training_samples]),
    dtype=torch.float32
)

Y = torch.tensor(
    np.array([sample[1] for sample in training_samples]),
    dtype=torch.float32
)


print("GNN Training")
print("=" * 40)

print("Input shape:", X.shape)
print("Target shape:", Y.shape)


# ----------------------------------------
# Create model
# ----------------------------------------

model = WeatherGNN()

loss_function = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ----------------------------------------
# Train model
# ----------------------------------------

epochs = 500

for epoch in range(epochs):

    predictions = model(X)

    loss = loss_function(
        predictions,
        Y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 50 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {loss.item():.6f}"
        )


# ----------------------------------------
# Final predictions
# ----------------------------------------

model.eval()

with torch.no_grad():

    predictions = model(X)


print()
print("Training completed!")
print("=" * 40)

for i in range(len(predictions)):

    actual = Y[i].numpy()
    predicted = predictions[i].numpy()

    print()
    print(f"Sample {i + 1}")
    print(
        f"Actual location: "
        f"({actual[0]:.2f}, {actual[1]:.2f})"
    )

    print(
        f"Predicted location: "
        f"({predicted[0]:.2f}, {predicted[1]:.2f})"
    )
