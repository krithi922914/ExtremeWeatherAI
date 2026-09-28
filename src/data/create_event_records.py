import numpy as np
from netCDF4 import Dataset

file_path = "data/samples/sample_weather.nc"

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


center_lats = []
center_lons = []
event_records = []


# ---------------------------------------
# STEP 1: Detect anomaly at each time
# ---------------------------------------

for t in range(wind_speed.shape[0]):

    anomaly_mask = wind_speed[t] >= threshold

    anomaly_indices = np.argwhere(anomaly_mask)

    if len(anomaly_indices) == 0:
        continue

    anomaly_lats = latitudes[anomaly_indices[:, 0]]
    anomaly_lons = longitudes[anomaly_indices[:, 1]]

    # Center
    center_lat = np.mean(anomaly_lats)
    center_lon = np.mean(anomaly_lons)

    # Bounding box
    min_lat = np.min(anomaly_lats)
    max_lat = np.max(anomaly_lats)

    min_lon = np.min(anomaly_lons)
    max_lon = np.max(anomaly_lons)

    # Maximum wind
    max_wind = np.max(wind_speed[t])

    # Intensity
    intensity = classify_intensity(max_wind)

    # Intensity score
    intensity_score = min((max_wind / 70) * 100, 100)

    center_lats.append(center_lat)
    center_lons.append(center_lon)

    record = {
        "time": t,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "south": min_lat,
        "north": max_lat,
        "west": min_lon,
        "east": max_lon,
        "max_wind": max_wind,
        "intensity_score": intensity_score,
        "intensity": intensity
    }

    event_records.append(record)


# ---------------------------------------
# STEP 2: Calculate movement
# ---------------------------------------

for i in range(len(event_records)):

    if i == 0:

        event_records[i]["movement_km"] = 0.0
        event_records[i]["direction"] = "Start"

        continue

    previous = event_records[i - 1]
    current = event_records[i]

    delta_lat = (
        current["center_lat"]
        - previous["center_lat"]
    )

    delta_lon = (
        current["center_lon"]
        - previous["center_lon"]
    )

    # Approximate latitude distance
    lat_distance = delta_lat * 111

    # Approximate longitude distance
    lon_distance = (
        delta_lon
        * 111
        * np.cos(np.radians(current["center_lat"]))
    )

    distance = np.sqrt(
        lat_distance ** 2
        + lon_distance ** 2
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

    if direction == "":
        direction = "Stationary"

    event_records[i]["movement_km"] = distance
    event_records[i]["direction"] = direction


# ---------------------------------------
# STEP 3: Display complete event records
# ---------------------------------------

print()
print("UNIFIED EXTREME WEATHER EVENT RECORDS")
print("=" * 60)

for event in event_records:

    print()
    print(f"Time: {event['time']}")

    print(
        f"Center: "
        f"({event['center_lat']:.2f}°N, "
        f"{event['center_lon']:.2f}°E)"
    )

    print(
        f"Bounding Box: "
        f"{event['south']:.1f}°N - "
        f"{event['north']:.1f}°N, "
        f"{event['west']:.1f}°E - "
        f"{event['east']:.1f}°E"
    )

    print(
        f"Maximum Wind: "
        f"{event['max_wind']:.2f} m/s"
    )

    print(
        f"Intensity Score: "
        f"{event['intensity_score']:.1f}/100"
    )

    print(
        f"Severity: "
        f"{event['intensity']}"
    )

    print(
        f"Movement: "
        f"{event['movement_km']:.2f} km"
    )

    print(
        f"Direction: "
        f"{event['direction']}"
    )