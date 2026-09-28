import json
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# EXTREME WEATHER AI - EMERGENCY ALERT GENERATOR
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_risk.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emergency_alert.json"
)


def load_risk_data():

    with open(RISK_FILE, "r") as file:
        return json.load(file)


def generate_alert(risk_data):

    risk_level = risk_data["risk_level"]

    latitude = risk_data["location"]["latitude"]
    longitude = risk_data["location"]["longitude"]

    weather = risk_data["weather"]

    max_wind = weather["maximum_wind_speed_mps"]
    intensity = weather["intensity_score"]
    movement = weather["movement_km"]
    direction = weather["direction"]

    if risk_level == "Extreme":

        status = "ACTIVE"

        recommended_action = (
            "Issue immediate extreme-weather alert and "
            "activate appropriate emergency response procedures."
        )

    elif risk_level == "Severe":

        status = "ACTIVE"

        recommended_action = (
            "Issue severe-weather warning and monitor "
            "the anomaly trajectory closely."
        )

    elif risk_level == "Moderate":

        status = "MONITORING"

        recommended_action = (
            "Continue monitoring the weather anomaly "
            "and update the forecast as new data arrives."
        )

    else:

        status = "NORMAL"

        recommended_action = (
            "No emergency action required. "
            "Continue routine monitoring."
        )

    alert = {

        "alert_id": "EWA-001",

        "alert_status": status,

        "event": {
            "type": "Extreme Weather Anomaly",
            "severity": risk_level
        },

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "location": {

            "latitude": latitude,

            "longitude": longitude,

            "affected_radius_km": 5
        },

        "weather": {

            "maximum_wind_speed_mps": max_wind,

            "intensity_score": intensity,

            "movement_km": movement,

            "direction": direction
        },

        "response": {

            "recommended_action": recommended_action
        }
    }

    return alert


def main():

    print("\n")
    print("=" * 60)
    print(" EXTREME WEATHER EMERGENCY ALERT GENERATOR")
    print("=" * 60)

    risk_data = load_risk_data()

    alert = generate_alert(risk_data)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as file:

        json.dump(
            alert,
            file,
            indent=4
        )

    print("\nAlert ID       :", alert["alert_id"])
    print("Status         :", alert["alert_status"])
    print("Severity       :", alert["event"]["severity"])

    print(
        "Location       : "
        f"{alert['location']['latitude']:.4f}, "
        f"{alert['location']['longitude']:.4f}"
    )

    print(
        "Affected Radius:",
        f"{alert['location']['affected_radius_km']} km"
    )

    print(
        "Maximum Wind   :",
        f"{alert['weather']['maximum_wind_speed_mps']:.2f} m/s"
    )

    print(
        "Intensity      :",
        f"{alert['weather']['intensity_score']:.2f}/100"
    )

    print(
        "Movement       :",
        f"{alert['weather']['movement_km']:.2f} km"
    )

    print(
        "Direction      :",
        alert["weather"]["direction"]
    )

    print(
        "Recommended    :",
        alert["response"]["recommended_action"]
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n")
    print("=" * 60)
    print(" EMERGENCY ALERT GENERATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()