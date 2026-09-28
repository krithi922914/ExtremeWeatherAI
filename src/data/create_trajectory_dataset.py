import numpy as np
import csv
from netCDF4 import Dataset

file_path = "data/samples/sample_weather.nc"
output_file = "data/processed/trajectory_dataset.csv"

dataset = Dataset(file_path, "r")

latitudes = dataset.variables["latitude"][:]
longitudes = dataset.variables["longitude"][:]
wind_speed = dataset.variables["wind_speed"][:]

dataset.close()

threshold = 40.0


def classify_intensity(wind):

    if wind < 20:
        return "Normal"
    elif wind < 40:
        return "Moderate"
    elif wind < 55:
        return "Severe"
    else:
        return "Extreme"


records = []

center_lats = []
center_lons = []


# ---------------------------------------
# Detect anomaly at every time step
# ---------------------------------------

for t in range(wind_speed.shape[0]):

    anomaly_mask = wind_speed[t] >= threshold
    anomaly_indices = np.argwhere(anomaly_mask)

    if len(anomaly_indices) == 0:
        continue

    anomaly_lats = latitudes[anomaly_indices[:, 0]]
    anomaly_lons = longitudes[anomaly_indices[:, 1]]

    center_lat = np.mean(anomaly_lats)
    center_lon = np.mean(anomaly_lons)

    max_wind = np.max(wind_speed[t])

    intensity_score = min((max_wind / 70) * 100, 100)

    intensity = classify_intensity(max_wind)

    center_lats.append(center_lat)
    center_lons.append(center_lon)

    records.append({
        "time": t,
        "latitude": center_lat,
        "longitude": center_lon,
        "max_wind": max_wind,
        "intensity_score": intensity_score,
        "intensity": intensity
    })


# ---------------------------------------
# Calculate movement
# ---------------------------------------

for i in range(len(records)):

    if i == 0:

        records[i]["movement_km"] = 0.0
        records[i]["direction"] = "Start"

        continue

    previous = records[i - 1]
    current = records[i]

    delta_lat = current["latitude"] - previous["latitude"]
    delta_lon = current["longitude"] - previous["longitude"]

    lat_distance = delta_lat * 111

    lon_distance = (
        delta_lon
        * 111
        * np.cos(np.radians(current["latitude"]))
    )

    distance = np.sqrt(
        lat_distance ** 2 +
        lon_distance ** 2
    )

    if delta_lat > 0:
        vertical = "North"
    elif delta_lat < 0:
        vertical = "South"
    else:
        vertical = ""

    if delta_lon > 0:
        horizontal = "East"
    elif delta_lon < 0:
        horizontal = "West"
    else:
        horizontal = ""

    direction = vertical + horizontal

    if direction == "":
        direction = "Stationary"

    records[i]["movement_km"] = distance
    records[i]["direction"] = direction


# ---------------------------------------
# Save CSV
# ---------------------------------------

fieldnames = [
    "time",
    "latitude",
    "longitude",
    "max_wind",
    "intensity_score",
    "intensity",
    "movement_km",
    "direction"
]

with open(output_file, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for record in records:
        writer.writerow(record)


print("Trajectory dataset created successfully!")
print()
print("File:", output_file)
print("Records:", len(records))
print("Features:", len(fieldnames))