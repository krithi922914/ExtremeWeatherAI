import csv
import numpy as np
import torch

from gnn_model import WeatherGNN


input_file = "data/processed/trajectory_dataset.csv"


# Load trajectory data
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


# Create the same samples used during training
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


X = torch.tensor(
    np.array([sample[0] for sample in training_samples]),
    dtype=torch.float32
)

Y = torch.tensor(
    np.array([sample[1] for sample in training_samples]),
    dtype=torch.float32
)


# Train a fresh model
model = WeatherGNN()

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


for epoch in range(500):

    predictions = model(X)

    loss = loss_function(
        predictions,
        Y
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Evaluate
model.eval()

with torch.no_grad():

    predictions = model(X)


errors = []

print("GNN Evaluation")
print("=" * 45)

for i in range(len(predictions)):

    actual = Y[i].numpy()
    predicted = predictions[i].numpy()

    error = np.sqrt(
        (actual[0] - predicted[0]) ** 2 +
        (actual[1] - predicted[1]) ** 2
    )

    errors.append(error)

    print(
        f"Sample {i + 1} | "
        f"Location Error: {error:.3f} degrees"
    )


print()
print("Evaluation Summary")
print("=" * 45)

print(f"Mean Error: {np.mean(errors):.3f} degrees")
print(f"Minimum Error: {np.min(errors):.3f} degrees")
print(f"Maximum Error: {np.max(errors):.3f} degrees")
