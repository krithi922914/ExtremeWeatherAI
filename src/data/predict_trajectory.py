import csv
import numpy as np

input_file = "data/processed/trajectory_dataset.csv"

records = []

# ---------------------------------------
# STEP 1: Load trajectory data
# ---------------------------------------

with open(input_file, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        records.append({
            "time": int(row["time"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"])
        })


# ---------------------------------------
# STEP 2: Predict next position
# ---------------------------------------

print("Baseline Extreme Weather Trajectory Prediction")
print("=" * 55)

for i in range(1, len(records) - 1):

    previous = records[i - 1]
    current = records[i]
    actual_next = records[i + 1]

    # Recent movement
    delta_lat = (
        current["latitude"]
        - previous["latitude"]
    )

    delta_lon = (
        current["longitude"]
        - previous["longitude"]
    )

    # Baseline prediction:
    # assume the anomaly continues moving
    # with the same recent displacement.

    predicted_lat = (
        current["latitude"] + delta_lat
    )

    predicted_lon = (
        current["longitude"] + delta_lon
    )

    # Prediction error
    lat_error = (
        predicted_lat
        - actual_next["latitude"]
    )

    lon_error = (
        predicted_lon
        - actual_next["longitude"]
    )

    error = np.sqrt(
        lat_error ** 2 +
        lon_error ** 2
    )

    print()
    print(f"From Time {current['time']}:")

    print(
        f"Predicted position: "
        f"({predicted_lat:.2f}°N, "
        f"{predicted_lon:.2f}°E)"
    )

    print(
        f"Actual position: "
        f"({actual_next['latitude']:.2f}°N, "
        f"{actual_next['longitude']:.2f}°E)"
    )

    print(
        f"Prediction error: "
        f"{error:.2f} degrees"
    )