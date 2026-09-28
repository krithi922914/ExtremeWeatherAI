import csv
import numpy as np

input_file = "data/processed/trajectory_dataset.csv"

records = []

with open(input_file, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        records.append({
            "time": int(row["time"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "wind": float(row["max_wind"]),
            "intensity": float(row["intensity_score"])
        })

training_samples = []

for i in range(len(records) - 2):

    previous = records[i]
    current = records[i + 1]
    target = records[i + 2]

    input_features = np.array([
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
    ])

    target_location = np.array([
        target["latitude"],
        target["longitude"]
    ])

    training_samples.append(
        (input_features, target_location)
    )

print("GNN Training Data Preparation")
print("=" * 45)

print()
print("Total training samples:", len(training_samples))

for i, (features, target) in enumerate(training_samples):

    print()
    print(f"Sample {i + 1}")
    print("Input shape:", features.shape)
    print("Input:")
    print(features)
    print("Target next location:", target)

X = np.array([
    sample[0]
    for sample in training_samples
])

Y = np.array([
    sample[1]
    for sample in training_samples
])

print()
print("FINAL DATA SHAPES")
print("-" * 45)
print("X shape:", X.shape)
print("Y shape:", Y.shape)
