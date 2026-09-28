import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

file_path = "data/samples/sample_weather.nc"

dataset = Dataset(file_path, "r")

latitudes = dataset.variables["latitude"][:]
longitudes = dataset.variables["longitude"][:]
wind_speed = dataset.variables["wind_speed"][:]

dataset.close()

# Choose one time step
time_step = 5

wind = wind_speed[time_step]

threshold = 40.0

# Find anomaly region
anomaly_mask = wind >= threshold
anomaly_indices = np.argwhere(anomaly_mask)

anomaly_lats = latitudes[anomaly_indices[:, 0]]
anomaly_lons = longitudes[anomaly_indices[:, 1]]

# Bounding box
min_lat = anomaly_lats.min()
max_lat = anomaly_lats.max()
min_lon = anomaly_lons.min()
max_lon = anomaly_lons.max()

# Create figure
plt.figure(figsize=(10, 7))

# Simple heatmap - does NOT require contourpy
plt.imshow(
    wind,
    origin="lower",
    extent=[
        longitudes.min(),
        longitudes.max(),
        latitudes.min(),
        latitudes.max()
    ],
    aspect="auto"
)

plt.colorbar(label="Wind Speed (m/s)")

# Plot anomaly points
plt.scatter(
    anomaly_lons,
    anomaly_lats,
    marker="o",
    label="Extreme Anomaly"
)

# Draw bounding box
plt.plot(
    [min_lon, max_lon, max_lon, min_lon, min_lon],
    [min_lat, min_lat, max_lat, max_lat, min_lat],
    linewidth=2,
    label="Dynamic Bounding Box"
)

plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.title(f"Extreme Weather Anomaly — Time Step {time_step}")

plt.legend()
plt.grid(True)

plt.show()