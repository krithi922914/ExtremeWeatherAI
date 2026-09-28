import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from spatiotemporal_gnn import SpatioTemporalWeatherGNN


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
# Create graph training samples
# ----------------------------------------

training_samples = []

for i in range(len(records) - 2):

    previous = records[i]
    current = records[i + 1]
    target = records[i + 2]

    node_features = np.array([
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
        (node_features, target_location)
    )


# ----------------------------------------
# Graph structure
# ----------------------------------------

# Node 0 -> Node 1
# Node 1 -> Node 0

edge_index = torch.tensor(
    [
        [0, 1],
        [1, 0]
    ],
    dtype=torch.long
).t()


# ----------------------------------------
# Create model
# ----------------------------------------

model = SpatioTemporalWeatherGNN()

loss_function = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


print("Spatio-Temporal GNN Training")
print("=" * 50)

print("Training samples:", len(training_samples))


# ----------------------------------------
# Training
# ----------------------------------------

epochs = 500

for epoch in range(epochs):

    total_loss = 0.0

    for node_features, target_location in training_samples:

        x = torch.tensor(
            node_features,
            dtype=torch.float32
        )

        y = torch.tensor(
            target_location,
            dtype=torch.float32
        ).unsqueeze(0)

        prediction = model(
            x,
            edge_index
        )

        loss = loss_function(
            prediction,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:

        average_loss = (
            total_loss / len(training_samples)
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {average_loss:.6f}"
        )


# ----------------------------------------
# Final evaluation
# ----------------------------------------

model.eval()

errors = []

print()
print("Training completed!")
print("=" * 50)

with torch.no_grad():

    for i, (node_features, target_location) in enumerate(
        training_samples
    ):

        x = torch.tensor(
            node_features,
            dtype=torch.float32
        )

        prediction = model(
            x,
            edge_index
        )

        predicted = prediction[0].numpy()

        actual = target_location

        error = np.sqrt(
            (actual[0] - predicted[0]) ** 2
            +
            (actual[1] - predicted[1]) ** 2
        )

        errors.append(error)

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

        print(
            f"Location error: "
            f"{error:.3f} degrees"
        )


print()
print("Evaluation Summary")
print("=" * 50)

print(
    f"Mean location error: "
    f"{np.mean(errors):.3f} degrees"
)

print(
    f"Minimum error: "
    f"{np.min(errors):.3f} degrees"
)

print(
    f"Maximum error: "
    f"{np.max(errors):.3f} degrees"
)

torch.save(
    model.state_dict(),
    "models/spatiotemporal_gnn.pth"
)

print()
print("Model saved successfully!")
print("File: models/spatiotemporal_gnn.pth")