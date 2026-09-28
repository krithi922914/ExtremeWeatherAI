import numpy as np
from netCDF4 import Dataset

file_path = "data/samples/sample_weather.nc"

dataset = Dataset(file_path, "r")

latitudes = dataset.variables["latitude"][:]
longitudes = dataset.variables["longitude"][:]
wind_speed = dataset.variables["wind_speed"][:]

print("Extreme Weather Detection")
print("=" * 30)

for t in range(wind_speed.shape[0]):

    max_index = np.unravel_index(
        np.argmax(wind_speed[t]),
        wind_speed[t].shape
    )

    lat_index, lon_index = max_index

    max_wind = wind_speed[t, lat_index, lon_index]
    max_lat = latitudes[lat_index]
    max_lon = longitudes[lon_index]

    print(
        f"Time {t}: "
        f"Wind = {max_wind:.2f} m/s | "
        f"Location = ({max_lat:.1f}°N, {max_lon:.1f}°E)"
    )

dataset.close()