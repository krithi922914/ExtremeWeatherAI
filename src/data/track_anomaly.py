import numpy as np
import matplotlib.pyplot as plt
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

print("Extreme Weather Trajectory")
print("=" * 40)

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

    print(
        f"Time {t}: "
        f"Center = ({center_lat:.2f}°N, {center_lon:.2f}°E)"
    )

# Plot trajectory
plt.figure(figsize=(10, 7))

plt.plot(
    center_lons,
    center_lats,
    marker="o",
    linewidth=2,
    label="Anomaly Trajectory"
)

# Label each time step
for t, (lon, lat) in enumerate(zip(center_lons, center_lats)):
    plt.text(
        lon,
        lat,
        f" T{t}",
        fontsize=10
    )

plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.title("Extreme Weather Anomaly Trajectory")

plt.grid(True)
plt.legend()

plt.show()