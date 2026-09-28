import numpy as np
from netCDF4 import Dataset

file_path = "data/samples/sample_weather.nc"

dataset = Dataset(file_path, "r")

wind_speed = dataset.variables["wind_speed"][:]

dataset.close()


def classify_intensity(wind):
    if wind < 20:
        return "Normal"
    elif wind < 40:
        return "Moderate"
    elif wind < 55:
        return "Severe"
    else:
        return "Extreme"


print("Extreme Weather Intensity Analysis")
print("=" * 45)

for t in range(wind_speed.shape[0]):

    max_wind = np.max(wind_speed[t])

    intensity = classify_intensity(max_wind)

    # Normalize wind speed into a 0–100 intensity score
    intensity_score = min((max_wind / 70) * 100, 100)

    print(
        f"Time {t} | "
        f"Maximum Wind: {max_wind:.2f} m/s | "
        f"Intensity Score: {intensity_score:.1f}/100 | "
        f"Level: {intensity}"
    )