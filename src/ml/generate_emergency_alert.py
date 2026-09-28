import json
import os
from datetime import datetime, timezone


# ---------------------------------------------------------
# Load existing risk classification
# ---------------------------------------------------------

input_file = "data/processed/weather_alert.json"

with open(input_file, "r") as file:
    alert = json.load(file)


# ---------------------------------------------------------
# Extract information
# ---------------------------------------------------------

latitude = alert["location"]["latitude"]
longitude = alert["location"]["longitude"]

wind_speed = alert["weather"]["maximum_wind_speed"]
intensity_score = alert["weather"]["intensity_score"]
movement = alert["weather"]["movement_km"]
direction = alert["weather"]["direction"]

risk_level = alert["risk"]["level"]


# ---------------------------------------------------------
# Prototype alert radius
# ---------------------------------------------------------

alert_radius_km = 5


# ---------------------------------------------------------
# Determine recommended action
# ---------------------------------------------------------

if risk_level == "Low":

    action = (
        "Continue routine monitoring. "
        "No immediate emergency action required."
    )

elif risk_level == "Moderate":

    action = (
        "Increase weather monitoring and prepare local response teams."
    )

elif risk_level == "Severe":

    action = (
        "Issue a regional warning and prepare emergency response resources."
    )

else:

    action = (
        "Issue immediate extreme-weather alert and activate "
        "appropriate emergency response procedures."
    )


# ---------------------------------------------------------
# Alert status
# ---------------------------------------------------------

if risk_level in ["Severe", "Extreme"]:
    alert_status = "ACTIVE"
else:
    alert_status = "MONITORING"


# ---------------------------------------------------------
# Create emergency alert
# ---------------------------------------------------------

emergency_alert = {

    "alert_id": "EWA-001",

    "alert_status": alert_status,

    "generated_at": datetime.now(timezone.utc).isoformat(),

    "event": {
        "type": "Extreme Weather Anomaly",
        "severity": risk_level
    },

    "location": {
        "latitude": latitude,
        "longitude": longitude,
        "affected_radius_km": alert_radius_km
    },

    "weather": {
        "maximum_wind_speed_mps": wind_speed,
        "intensity_score": intensity_score,
        "movement_km": movement,
        "direction": direction
    },

    "response": {
        "recommended_action": action
    }

}


# ---------------------------------------------------------
# Save emergency alert
# ---------------------------------------------------------

output_directory = "data/processed"

os.makedirs(output_directory, exist_ok=True)

output_file = os.path.join(
    output_directory,
    "emergency_alert.json"
)

with open(output_file, "w") as file:
    json.dump(
        emergency_alert,
        file,
        indent=4
    )


# ---------------------------------------------------------
# Display alert
# ---------------------------------------------------------

print()
print("=" * 65)
print("EXTREME WEATHER EMERGENCY ALERT")
print("=" * 65)

print()

print(f"Alert ID       : {emergency_alert['alert_id']}")
print(f"Status         : {alert_status}")
print(f"Severity       : {risk_level}")

print()

print("LOCATION")
print("-" * 65)
print(f"Latitude       : {latitude}")
print(f"Longitude      : {longitude}")
print(f"Affected Radius: {alert_radius_km} km")

print()

print("WEATHER")
print("-" * 65)
print(f"Maximum Wind   : {wind_speed} m/s")
print(f"Intensity      : {intensity_score}/100")
print(f"Movement       : {movement} km")
print(f"Direction      : {direction}")

print()

print("RECOMMENDED ACTION")
print("-" * 65)
print(action)

print()

print("=" * 65)
print("Emergency alert generated successfully!")
print(f"Saved to: {output_file}")
print("=" * 65)