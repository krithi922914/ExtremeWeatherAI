import numpy as np
from netCDF4 import Dataset

file_path = "data/samples/sample_weather.nc"

dataset = Dataset(file_path, "r")

latitudes = dataset.variables["latitude"][:]
longitudes = dataset.variables["longitude"][:]
wind_speed = dataset.variables["wind_speed"][:]

threshold = 40.0

print("Dynamic Extreme Weather Bounding Boxes")
print("=" * 50)

for t in range(wind_speed.shape[0]):

    anomaly_mask = wind_speed[t] >= threshold
    anomaly_indices = np.argwhere(anomaly_mask)

    if len(anomaly_indices) == 0:
        print(f"Time {t}: No anomaly detected")
        continue

    anomaly_lats = latitudes[anomaly_indices[:, 0]]
    anomaly_lons = longitudes[anomaly_indices[:, 1]]

    min_lat = anomaly_lats.min()
    max_lat = anomaly_lats.max()
    min_lon = anomaly_lons.min()
    max_lon = anomaly_lons.max()

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    print(f"\nTime {t}")
    print(f"Bounding Box:")
    print(f"  South: {min_lat:.1f}°N")
    print(f"  North: {max_lat:.1f}°N")
    print(f"  West : {min_lon:.1f}°E")
    print(f"  East : {max_lon:.1f}°E")
    print(f"Center: ({center_lat:.1f}°N, {center_lon:.1f}°E)")
    print(f"Anomaly points: {len(anomaly_indices)}")

dataset.close()