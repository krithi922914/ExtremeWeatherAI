import json
from pathlib import Path


# ============================================================
# EXTREME WEATHER AI - RISK CLASSIFICATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAJECTORY_FILE = (
    PROJECT_ROOT / "data" / "processed" / "trajectory_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "weather_risk.json"
)


def calculate_intensity_score(max_wind):

    score = (max_wind / 70.0) * 100.0

    return min(score, 100.0)


def classify_risk(max_wind):

    if max_wind >= 55:
        return "Extreme"

    elif max_wind >= 40:
        return "Severe"

    elif max_wind >= 20:
        return "Moderate"

    else:
        return "Normal"


def load_latest_weather():

    import csv

    with open(TRAJECTORY_FILE, "r") as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:
        raise ValueError("Trajectory dataset is empty.")

    return rows[-1]


def main():

    latest = load_latest_weather()

    latitude = float(latest["latitude"])
    longitude = float(latest["longitude"])
    max_wind = float(latest["max_wind"])

    movement = float(latest["movement_km"])
    direction = latest["direction"]

    intensity_score = calculate_intensity_score(max_wind)

    risk_level = classify_risk(max_wind)

    result = {

        "risk_level": risk_level,

        "location": {
            "latitude": latitude,
            "longitude": longitude
        },

        "weather": {
            "maximum_wind_speed_mps": max_wind,
            "intensity_score": round(intensity_score, 2),
            "movement_km": movement,
            "direction": direction
        }

    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print("\n")
    print("=" * 60)
    print(" EXTREME WEATHER RISK CLASSIFICATION")
    print("=" * 60)

    print(f"\nLatitude        : {latitude:.4f}")
    print(f"Longitude       : {longitude:.4f}")
    print(f"Maximum Wind    : {max_wind:.2f} m/s")
    print(f"Intensity Score : {intensity_score:.2f}/100")
    print(f"Movement        : {movement:.2f} km")
    print(f"Direction       : {direction}")
    print(f"Risk Level      : {risk_level}")

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()