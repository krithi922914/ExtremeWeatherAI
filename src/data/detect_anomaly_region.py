import numpy as np
from netCDF4 import Dataset

file_path = "data/samples/sample_weather.nc"

dataset = Dataset(file_path, "r")

latitudes = dataset.variables["latitude"][:]
longitudes = dataset.variables["longitude"][:]
wind_speed = dataset.variables["wind_speed"][:]

print("Extreme Weather Anomaly Region Detection")
print("=" * 45)

threshold = 40.0  # m/s

for t in range(wind_speed.shape[0]):

    anomaly_mask = wind_speed[t] >= threshold

    anomaly_indices = np.argwhere(anomaly_mask)

    print(f"\nTime {t}")
    print("-" * 20)

    print("Anomaly points:", len(anomaly_indices))

    if len(anomaly_indices) > 0:

        anomaly_lats = latitudes[anomaly_indices[:, 0]]
        anomaly_lons = longitudes[anomaly_indices[:, 1]]

        print(
            f"Latitude range: "
            f"{anomaly_lats.min():.1f}°N - "
            f"{anomaly_lats.max():.1f}°N"
        )

        print(
            f"Longitude range: "
            f"{anomaly_lons.min():.1f}°E - "
            f"{anomaly_lons.max():.1f}°E"
        )

dataset.close()