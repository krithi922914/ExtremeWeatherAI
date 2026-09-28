from flask import Flask, render_template, jsonify
import json
import csv
import os

print("APP.PY HAS STARTED")

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ALERT_FILE = os.path.join(
    PROCESSED_DIR,
    "emergency_alert.json"
)

RISK_FILE = os.path.join(
    PROCESSED_DIR,
    "weather_risk.json"
)

TRAJECTORY_FILE = os.path.join(
    PROCESSED_DIR,
    "trajectory_dataset.csv"
)

PIPELINE_STATUS_FILE = os.path.join(
    PROCESSED_DIR,
    "pipeline_status.json"
)

# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "templates"
    )
)

# ============================================================
# JSON LOADER
# ============================================================

def load_json(file_path, default_data=None):

    if not os.path.exists(file_path):

        print(f"File not found: {file_path}")

        if default_data is not None:
            return default_data

        return {}

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as error:

        print(
            f"Error reading {file_path}: {error}"
        )

        return default_data or {}


# ============================================================
# ALERT
# ============================================================

def load_alert():

    return load_json(
        ALERT_FILE,
        {
            "alert_id": "N/A",
            "status": "NO DATA",
            "severity": "Unknown",

            "location": {
                "latitude": 0,
                "longitude": 0,
                "affected_radius_km": 0
            },

            "maximum_wind_speed": 0,
            "intensity_score": 0,
            "movement_km": 0,
            "direction": "Unknown",

            "response": {
                "recommended_action":
                    "No alert data available."
            }
        }
    )


# ============================================================
# RISK
# ============================================================

def load_risk():

    return load_json(
        RISK_FILE,
        {}
    )


# ============================================================
# TRAJECTORY
# ============================================================

def load_trajectory():

    trajectory = []

    if not os.path.exists(
        TRAJECTORY_FILE
    ):
        print(
            f"Trajectory file not found: "
            f"{TRAJECTORY_FILE}"
        )

        return trajectory

    try:

        with open(
            TRAJECTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                trajectory.append(
                    {
                        "latitude":
                            float(row["latitude"]),

                        "longitude":
                            float(row["longitude"]),

                        "wind":
                            float(row["max_wind"])
                    }
                )

    except Exception as error:

        print(
            f"Error reading trajectory: {error}"
        )

    return trajectory


# ============================================================
# PIPELINE STATUS
# ============================================================

def load_pipeline_status():

    default_status = {

        "stages": [],

        "completed": 0,

        "total": 9,

        "percentage": 0
    }

    return load_json(
        PIPELINE_STATUS_FILE,
        default_status
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    alert = load_alert()

    risk = load_risk()

    trajectory = load_trajectory()

    pipeline_status = load_pipeline_status()

    return render_template(

        "dashboard.html",

        alert=alert,

        risk=risk,

        trajectory=trajectory,

        pipeline_status=pipeline_status
    )


# ============================================================
# API — ALERT
# ============================================================

@app.route("/api/alert")
def api_alert():

    return jsonify(
        load_alert()
    )


# ============================================================
# API — RISK
# ============================================================

@app.route("/api/risk")
def api_risk():

    return jsonify(
        load_risk()
    )


# ============================================================
# API — TRAJECTORY
# ============================================================

@app.route("/api/trajectory")
def api_trajectory():

    return jsonify(
        load_trajectory()
    )


# ============================================================
# API — PIPELINE STATUS
# ============================================================

@app.route("/api/pipeline-status")
def api_pipeline_status():

    return jsonify(
        load_pipeline_status()
    )


# ============================================================
# API — HEALTH
# ============================================================

@app.route("/api/health")
def api_health():

    return jsonify(
        {
            "status": "online",

            "pipeline_status":
                load_pipeline_status()
        }
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("EXTREME WEATHER AI DASHBOARD")
    print("=" * 60)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Pipeline status: "
        f"{PIPELINE_STATUS_FILE}"
    )

    print()

    print(
        "Dashboard available at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )