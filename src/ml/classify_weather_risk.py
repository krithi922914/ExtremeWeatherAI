import csv
import json
import os


# ---------------------------------------------------------
# Prototype risk thresholds
# ---------------------------------------------------------

def classify_risk(wind_speed):
    if wind_speed < 20:
        return "Low"
    elif wind_speed < 40:
        return "Moderate"
    elif wind_speed < 55:
        return "Severe"
    else:
        return "Extreme"


def get_risk_description(risk):
    descriptions = {
        "Low": "No significant extreme-weather signal detected.",
        "Moderate": "Moderate weather anomaly detected. Monitoring recommended.",
        "Severe": "Severe weather anomaly detected. Alert recommended.",
        "Extreme": "Extreme weather anomaly detected. Immediate alert recommended."
    }

    return descriptions[risk]


# ---------------------------------------------------------
# Load latest weather record
# ---------------------------------------------------------

dataset_path = "data/processed/trajectory_dataset.csv"

with open(dataset_path, "r") as file:
    reader = csv.DictReader(file)
    records = list(reader)


latest = records[-1]

latitude = float(latest["latitude"])
longitude = float(latest["longitude"])
wind_speed = float(latest["max_wind"])
intensity_score = float(latest["intensity_score"])
direction = latest["direction"]
movement = float(latest["movement_km"])


# ---------------------------------------------------------
# Classify risk
# ---------------------------------------------------------

risk_level = classify_risk(wind_speed)
description = get_risk_description(risk_level)


# ---------------------------------------------------------
# Create alert record
# ---------------------------------------------------------

alert = {
    "location": {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4)
    },

    "weather": {
        "maximum_wind_speed": round(wind_speed, 2),
        "intensity_score": round(intensity_score, 2),
        "movement_km": round(movement, 2),
        "direction": direction
    },

    "risk": {
        "level": risk_level,
        "description": description
    }
}


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

print()
print("=" * 60)
print("EXTREME WEATHER RISK CLASSIFICATION")
print("=" * 60)

print()

print("Location")
print("-" * 60)
print(f"Latitude  : {latitude:.4f}")
print(f"Longitude : {longitude:.4f}")

print()

print("Weather Information")
print("-" * 60)
print(f"Maximum Wind Speed : {wind_speed:.2f} m/s")
print(f"Intensity Score    : {intensity_score:.2f}")
print(f"Movement           : {movement:.2f} km")
print(f"Direction          : {direction}")

print()

print("Risk Assessment")
print("-" * 60)
print(f"Risk Level         : {risk_level}")
print(f"Description        : {description}")

print()

# ---------------------------------------------------------
# Save alert as JSON
# ---------------------------------------------------------

output_directory = "data/processed"

os.makedirs(output_directory, exist_ok=True)

output_file = os.path.join(
    output_directory,
    "weather_alert.json"
)

with open(output_file, "w") as file:
    json.dump(alert, file, indent=4)


print("=" * 60)
print("Alert generated successfully!")
print(f"Saved to: {output_file}")
print("=" * 60)