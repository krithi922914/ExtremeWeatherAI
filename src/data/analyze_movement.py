import numpy as np
from netCDF4 import Dataset

file_path = "data/samples/sample_weather.nc"

dataset = Dataset(file_path, "r")

latitudes = dataset.variables["latitude"][:]
longitudes = dataset.variables["longitude"][:]
wind_speed = dataset.variables["wind_speed"][:]

dataset.close()

threshold = 40.0

center_lats = []
center_lons = []

# Find anomaly center at every time step
for t in range(wind_speed.shape[0]):

    anomaly_mask = wind_speed[t] >= threshold
    anomaly_indices = np.argwhere(anomaly_mask)

    if len(anomaly_indices) == 0:
        continue

    anomaly_lats = latitudes[anomaly_indices[:, 0]]
    anomaly_lons = longitudes[anomaly_indices[:, 1]]

    center_lat = np.mean(anomaly_lats)
    center_lon = np.mean(anomaly_lons)

    center_lats.append(center_lat)
    center_lons.append(center_lon)


print("Extreme Weather Movement Analysis")
print("=" * 45)

# Analyze movement between consecutive time steps
for i in range(1, len(center_lats)):

    previous_lat = center_lats[i - 1]
    previous_lon = center_lons[i - 1]

    current_lat = center_lats[i]
    current_lon = center_lons[i]

    # Movement in latitude and longitude
    delta_lat = current_lat - previous_lat
    delta_lon = current_lon - previous_lon

    # Approximate distance
    # 1 degree latitude ≈ 111 km
    lat_distance = delta_lat * 111

    # Longitude distance changes with latitude
    lon_distance = (
        delta_lon
        * 111
        * np.cos(np.radians(current_lat))
    )

    distance = np.sqrt(
        lat_distance ** 2 +
        lon_distance ** 2
    )

    # Direction
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

    print(
        f"T{i-1} → T{i} | "
        f"Movement: {distance:.2f} km | "
        f"Direction: {direction}"
    )