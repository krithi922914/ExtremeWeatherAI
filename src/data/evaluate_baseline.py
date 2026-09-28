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
            "longitude": float(row["longitude"])
        })


errors = []

print("Baseline Trajectory Evaluation")
print("=" * 45)

for i in range(1, len(records) - 1):

    previous = records[i - 1]
    current = records[i]
    actual_next = records[i + 1]

    # Predict using recent movement
    delta_lat = current["latitude"] - previous["latitude"]
    delta_lon = current["longitude"] - previous["longitude"]

    predicted_lat = current["latitude"] + delta_lat
    predicted_lon = current["longitude"] + delta_lon

    # Calculate error
    lat_error = predicted_lat - actual_next["latitude"]
    lon_error = predicted_lon - actual_next["longitude"]

    error = np.sqrt(
        lat_error ** 2 +
        lon_error ** 2
    )

    errors.append(error)

    print(
        f"Prediction {i}: "
        f"Error = {error:.3f} degrees"
    )


# ---------------------------------------
# Overall evaluation
# ---------------------------------------

mean_error = np.mean(errors)
maximum_error = np.max(errors)
minimum_error = np.min(errors)

print()
print("OVERALL RESULTS")
print("-" * 45)

print(f"Average Error : {mean_error:.3f} degrees")
print(f"Minimum Error : {minimum_error:.3f} degrees")
print(f"Maximum Error : {maximum_error:.3f} degrees")
print(f"Predictions   : {len(errors)}")