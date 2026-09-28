import numpy as np
from netCDF4 import Dataset


# --------------------------------
# 1. Define our weather grid
# --------------------------------

latitudes = np.arange(5, 26, 1)
longitudes = np.arange(75, 101, 1)

time_steps = 10


# --------------------------------
# 2. Create random sample weather
# --------------------------------

rng = np.random.default_rng(42)

wind_speed = (
    5
    + 15 * rng.random(
        (time_steps, len(latitudes), len(longitudes))
    )
)


# --------------------------------
# 3. Create an artificial
#    extreme-weather anomaly
# --------------------------------

for t in range(time_steps):

    center_lat = 12 + (t * 0.7)
    center_lon = 82 + (t * 0.8)

    for i, lat in enumerate(latitudes):

        for j, lon in enumerate(longitudes):

            distance = np.sqrt(
                (lat - center_lat) ** 2
                + (lon - center_lon) ** 2
            )

            anomaly = 50 * np.exp(
                -(distance ** 2) / 5
            )

            wind_speed[t, i, j] += anomaly


# --------------------------------
# 4. Create NetCDF file
# --------------------------------

output_file = "data/samples/sample_weather.nc"

dataset = Dataset(
    output_file,
    "w",
    format="NETCDF4"
)


# --------------------------------
# 5. Create dimensions
# --------------------------------

dataset.createDimension(
    "time",
    time_steps
)

dataset.createDimension(
    "latitude",
    len(latitudes)
)

dataset.createDimension(
    "longitude",
    len(longitudes)
)


# --------------------------------
# 6. Create variables
# --------------------------------

lat = dataset.createVariable(
    "latitude",
    "f4",
    ("latitude",)
)

lon = dataset.createVariable(
    "longitude",
    "f4",
    ("longitude",)
)

time = dataset.createVariable(
    "time",
    "i4",
    ("time",)
)

wind = dataset.createVariable(
    "wind_speed",
    "f4",
    ("time", "latitude", "longitude")
)


# --------------------------------
# 7. Store data
# --------------------------------

lat[:] = latitudes
lon[:] = longitudes
time[:] = np.arange(time_steps)

wind[:] = wind_speed


# --------------------------------
# 8. Add information
# --------------------------------

wind.units = "m/s"
wind.description = "Sample wind speed"


# --------------------------------
# 9. Close the dataset
# --------------------------------

dataset.close()


print("Weather dataset created successfully!")
print()
print("File:", output_file)
print("Latitude points:", len(latitudes))
print("Longitude points:", len(longitudes))
print("Time steps:", time_steps)