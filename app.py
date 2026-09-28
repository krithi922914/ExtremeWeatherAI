from flask import Flask, jsonify, render_template_string
from pathlib import Path
import json
import csv
import math

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"


# ============================================================
# DEFAULT / FALLBACK DATA
# ============================================================

DEFAULT_ALERT = {
    "alert_id": "EWA-001",
    "status": "ACTIVE",
    "severity": "Extreme",
    "latitude": 18.0909,
    "longitude": 89.3636,
    "affected_radius_km": 5,
    "maximum_wind_ms": 62.91,
    "intensity_score": 89.87,
    "movement_km": 103.90,
    "direction": "NorthEast"
}


DEFAULT_TRAJECTORY = [
    {
        "time": "T0",
        "latitude": 12.00,
        "longitude": 82.00,
        "max_wind": 59.99,
        "intensity": "Extreme",
        "movement_km": 0,
        "direction": "Start"
    },
    {
        "time": "T1",
        "latitude": 12.82,
        "longitude": 82.82,
        "max_wind": 55.36,
        "intensity": "Extreme",
        "movement_km": 126.85,
        "direction": "NorthEast"
    },
    {
        "time": "T2",
        "latitude": 13.18,
        "longitude": 83.45,
        "max_wind": 66.08,
        "intensity": "Extreme",
        "movement_km": 79.74,
        "direction": "NorthEast"
    },
    {
        "time": "T3",
        "latitude": 14.25,
        "longitude": 84.33,
        "max_wind": 62.88,
        "intensity": "Extreme",
        "movement_km": 151.65,
        "direction": "NorthEast"
    },
    {
        "time": "T4",
        "latitude": 14.62,
        "longitude": 85.38,
        "max_wind": 63.58,
        "intensity": "Extreme",
        "movement_km": 119.98,
        "direction": "NorthEast"
    },
    {
        "time": "T5",
        "latitude": 15.64,
        "longitude": 86.09,
        "max_wind": 65.46,
        "intensity": "Extreme",
        "movement_km": 136.17,
        "direction": "NorthEast"
    },
    {
        "time": "T6",
        "latitude": 16.20,
        "longitude": 86.70,
        "max_wind": 59.04,
        "intensity": "Extreme",
        "movement_km": 90.16,
        "direction": "NorthEast"
    },
    {
        "time": "T7",
        "latitude": 17.00,
        "longitude": 88.00,
        "max_wind": 63.68,
        "intensity": "Extreme",
        "movement_km": 164.10,
        "direction": "NorthEast"
    },
    {
        "time": "T8",
        "latitude": 17.50,
        "longitude": 88.60,
        "max_wind": 62.88,
        "intensity": "Extreme",
        "movement_km": 84.35,
        "direction": "NorthEast"
    },
    {
        "time": "T9",
        "latitude": 18.0909,
        "longitude": 89.3636,
        "max_wind": 62.91,
        "intensity": "Extreme",
        "movement_km": 103.90,
        "direction": "NorthEast"
    }
]


DEFAULT_GNN = {
    "model": "Spatio-Temporal GNN",
    "current_latitude": 18.0909,
    "current_longitude": 89.3636,
    "predicted_latitude": 18.97,
    "predicted_longitude": 90.10
}


DEFAULT_DOWNSCALING = {
    "status": "completed",
    "prototype": True,
    "input_resolution": "10 x 10",
    "output_resolution": "20 x 20",
    "dataset_samples": 200,
    "models": {
        "standard_cnn": {
            "name": "Standard CNN",
            "mse": 0.548185,
            "mae": 0.589521,
            "mean_max_error": 0.621413,
            "median_max_error": 0.523083,
            "extreme_pixel_mae": 0.638055
        },
        "extreme_cnn": {
            "name": "Extreme-Aware CNN",
            "mse": 0.698831,
            "mae": 0.661480,
            "mean_max_error": 0.521488,
            "median_max_error": 0.419609,
            "extreme_pixel_mae": 0.587120,
            "max_preservation_better_samples": 118,
            "max_preservation_percentage": 59.0
        },
        "physics_extreme_cnn": {
            "name": "Physics + Extreme CNN",
            "mse": 0.717017,
            "mae": 0.672071,
            "mean_max_error": 0.563334,
            "median_max_error": 0.469994,
            "extreme_pixel_mae": 0.622835
        }
    }
}


# ============================================================
# FILE HELPERS
# ============================================================

def read_json(filename, fallback):
    path = DATA / filename

    try:
        if not path.exists():
            return fallback

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return fallback


def read_csv(filename):
    path = DATA / filename

    try:
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    except Exception:
        return []


def number(value, fallback=0.0):
    try:
        return float(value)
    except Exception:
        return fallback


# ============================================================
# DATA LOADERS
# ============================================================

def get_alert():
    data = read_json("emergency_alert.json", {})

    result = DEFAULT_ALERT.copy()

    if isinstance(data, dict):
        result.update(data)

    return result


def get_trajectory():
    rows = read_csv("trajectory_dataset.csv")

    if not rows:
        return DEFAULT_TRAJECTORY

    output = []

    for index, row in enumerate(rows[:20]):

        fallback = DEFAULT_TRAJECTORY[
            min(index, len(DEFAULT_TRAJECTORY) - 1)
        ]

        wind = number(
            row.get("max_wind"),
            number(row.get("wind"), fallback["max_wind"])
        )

        intensity = row.get("intensity")

        if not intensity:
            if wind >= 55:
                intensity = "Extreme"
            elif wind >= 40:
                intensity = "Severe"
            elif wind >= 20:
                intensity = "Moderate"
            else:
                intensity = "Normal"

        output.append({
            "time": row.get("time", fallback["time"]),
            "latitude": number(
                row.get("latitude"),
                fallback["latitude"]
            ),
            "longitude": number(
                row.get("longitude"),
                fallback["longitude"]
            ),
            "max_wind": wind,
            "intensity": intensity,
            "movement_km": number(
                row.get("movement_km"),
                fallback.get("movement_km", 0)
            ),
            "direction": row.get(
                "direction",
                fallback.get("direction", "NorthEast")
            )
        })

    return output


def get_gnn():
    data = read_json(
        "gnn_prediction.json",
        DEFAULT_GNN
    )

    result = DEFAULT_GNN.copy()

    if isinstance(data, dict):
        result.update(data)

    return result


def get_downscaling():
    data = read_json(
        "downscaling_results.json",
        DEFAULT_DOWNSCALING
    )

    if not isinstance(data, dict):
        return DEFAULT_DOWNSCALING

    return data


# ============================================================
# DISTANCE / LOCATION RISK
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):
    radius = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return radius * c


def distance_score(distance):
    if distance <= 5:
        return 100

    if distance <= 25:
        return 85

    if distance <= 75:
        return 60

    if distance <= 150:
        return 30

    return 10


def calculate_location_risk(latitude, longitude):

    alert = get_alert()
    gnn = get_gnn()

    current_distance = haversine(
        latitude,
        longitude,
        number(
            alert.get("latitude"),
            DEFAULT_ALERT["latitude"]
        ),
        number(
            alert.get("longitude"),
            DEFAULT_ALERT["longitude"]
        )
    )

    predicted_distance = haversine(
        latitude,
        longitude,
        number(
            gnn.get("predicted_latitude"),
            DEFAULT_GNN["predicted_latitude"]
        ),
        number(
            gnn.get("predicted_longitude"),
            DEFAULT_GNN["predicted_longitude"]
        )
    )

    current_distance_score = distance_score(
        current_distance
    )

    predicted_distance_score = distance_score(
        predicted_distance
    )

    intensity = number(
        alert.get("intensity_score"),
        89.87
    )

    movement = number(
        alert.get("movement_km"),
        103.90
    )

    movement_score = min(
        (movement / 200.0) * 100.0,
        100.0
    )

    final_score = (
        current_distance_score * 0.35
        +
        predicted_distance_score * 0.30
        +
        intensity * 0.25
        +
        movement_score * 0.10
    )

    if final_score >= 80:
        level = "Extreme"
        message = (
            "Your location is within the highest "
            "prototype risk category."
        )

    elif final_score >= 60:
        level = "High"
        message = (
            "Your location is in a high prototype "
            "risk zone."
        )

    elif final_score >= 40:
        level = "Moderate"
        message = (
            "Your location has moderate prototype "
            "exposure."
        )

    elif final_score >= 20:
        level = "Low"
        message = (
            "Your location has relatively low "
            "prototype exposure."
        )

    else:
        level = "Minimal"
        message = (
            "Your location is currently outside "
            "the main prototype risk region."
        )

    return {
        "user_latitude": latitude,
        "user_longitude": longitude,

        "current_distance_km": round(
            current_distance,
            2
        ),

        "predicted_distance_km": round(
            predicted_distance,
            2
        ),

        "current_event": {
            "latitude": alert.get("latitude"),
            "longitude": alert.get("longitude"),
            "maximum_wind_ms": alert.get(
                "maximum_wind_ms"
            ),
            "intensity_score": alert.get(
                "intensity_score"
            )
        },

        "predicted_event": {
            "latitude": gnn.get(
                "predicted_latitude"
            ),
            "longitude": gnn.get(
                "predicted_longitude"
            )
        },

        "movement_km": movement,
        "direction": alert.get(
            "direction",
            "NorthEast"
        ),

        "risk_score": round(
            final_score,
            2
        ),

        "risk_level": level,
        "message": message
    }


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
def home():

    return render_template_string(
        PAGE,
        alert=get_alert(),
        trajectory=get_trajectory(),
        gnn=get_gnn(),
        downscaling=get_downscaling()
    )


@app.get("/api/health")
def api_health():

    return jsonify({
        "status": "healthy",
        "project": "ExtremeWeatherAI",
        "mode": "Prototype"
    })


@app.get("/api/dashboard")
def api_dashboard():

    return jsonify({
        "alert": get_alert(),
        "trajectory": get_trajectory(),
        "gnn": get_gnn(),
        "downscaling": get_downscaling()
    })


@app.get("/api/alert")
def api_alert():

    return jsonify(get_alert())


@app.get("/api/trajectory")
def api_trajectory():

    return jsonify(get_trajectory())


@app.get("/api/gnn")
def api_gnn():

    return jsonify(get_gnn())


@app.get("/api/downscaling")
def api_downscaling():

    return jsonify(get_downscaling())


@app.get("/api/risk")
def api_risk():

    return jsonify({
        "risk_level": get_alert().get(
            "severity",
            "Extreme"
        ),
        "risk_score": get_alert().get(
            "intensity_score",
            89.87
        )
    })


@app.get("/api/health")
def api_health_duplicate():

    return jsonify({
        "status": "healthy",
        "project": "ExtremeWeatherAI",
        "mode": "Prototype"
    })


@app.get("/api/pipeline-status")
def api_pipeline():

    stages = [
        "Weather Data Ingestion",
        "Extreme Anomaly Detection",
        "Anomaly Region Extraction",
        "Dynamic Bounding Boxes",
        "Weather Event Records",
        "Trajectory Dataset",
        "Spatio-Temporal Graph",
        "AI Trajectory Prediction",
        "Risk Classification",
        "Emergency Alert Generation"
    ]

    return jsonify({
        "completion": 100,
        "stages": len(stages),
        "total": len(stages),
        "status": "ONLINE",
        "items": [
            {
                "id": index + 1,
                "name": name,
                "status": "COMPLETED"
            }
            for index, name in enumerate(stages)
        ]
    })


@app.get("/api/location/<latitude>/<longitude>")
def api_location(latitude, longitude):

    try:
        lat = float(latitude)
        lon = float(longitude)

        result = calculate_location_risk(
            lat,
            lon
        )

        return jsonify(result)

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 400


# ============================================================
# FRONTEND
# ============================================================

PAGE = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>ExtremeWeather AI</title>

<style>

:root {

    --bg: #050a12;
    --panel: #0b1422;
    --panel2: #0e1b2d;
    --line: #1b3148;

    --text: #e9f3ff;
    --muted: #8ea4bc;

    --blue: #28a9ff;
    --cyan: #4de5ff;
    --red: #ff5063;
    --orange: #ff9c42;
    --yellow: #ffd45a;
    --green: #39df92;
    --purple: #a87cff;

}

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    background:
        radial-gradient(
            circle at 75% 0%,
            #102b47 0,
            #050a12 40%
        );

    color: var(--text);

    font-family:
        Segoe UI,
        Arial,
        sans-serif;

}

button,
input {

    font-family: inherit;

}

button {

    cursor: pointer;

}

.app {

    min-height: 100vh;

    display: flex;

}

.sidebar {

    position: fixed;

    inset: 0 auto 0 0;

    width: 245px;

    background: #050b13ee;

    border-right: 1px solid var(--line);

    padding: 22px 14px;

    z-index: 10;

}

.brand {

    font-size: 17px;

    font-weight: 900;

    letter-spacing: .14em;

    padding: 8px;

}

.brand span {

    display: block;

    color: var(--blue);

    font-size: 21px;

}

.system-online {

    margin: 20px 8px;

    padding: 9px;

    border: 1px solid #17563f;

    border-radius: 8px;

    color: var(--green);

    font-size: 11px;

}

.nav {

    display: grid;

    gap: 4px;

}

.nav button {

    border: 0;

    background: transparent;

    color: #9fb2c8;

    text-align: left;

    padding: 11px;

    border-radius: 8px;

}

.nav button:hover,
.nav button.active {

    background: #11243a;

    color: white;

}

.main {

    margin-left: 245px;

    width: calc(100% - 245px);

    padding: 26px 32px;

}

.topbar {

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 20px;

    margin-bottom: 22px;

}

.eyebrow {

    color: var(--blue);

    font-size: 10px;

    letter-spacing: .17em;

}

h1 {

    margin: 6px 0;

    font-size: 31px;

}

.subtitle {

    color: var(--muted);

}

.actions {

    display: flex;

    gap: 7px;

    flex-wrap: wrap;

}

.badge,
.btn {

    border: 1px solid var(--line);

    background: #0c192a;

    color: #d8e8f8;

    border-radius: 8px;

    padding: 8px 10px;

    font-size: 11px;

}

.btn:hover {

    border-color: #2878ad;

}

.primary {

    background: #075987;

    border-color: #138bd0;

}

.cards {

    display: grid;

    grid-template-columns:
        repeat(6, 1fr);

    gap: 10px;

}

.card {

    background:
        linear-gradient(
            145deg,
            #0d1929f5,
            #09121ef5
        );

    border: 1px solid var(--line);

    border-radius: 12px;

    padding: 16px;

    box-shadow:
        0 14px 35px #0005;

}

.label {

    color: var(--muted);

    font-size: 11px;

}

.value {

    font-size: 23px;

    font-weight: 800;

    margin: 8px 0;

}

.small {

    color: var(--muted);

    font-size: 11px;

}

.red {

    color: var(--red);

}

.blue {

    color: var(--blue);

}

.green {

    color: var(--green);

}

.yellow {

    color: var(--yellow);

}

.orange {

    color: var(--orange);

}

.purple {

    color: var(--purple);

}

.section {

    margin-top: 14px;

}

.grid2 {

    display: grid;

    grid-template-columns:
        1.65fr 1fr;

    gap: 14px;

}

.grid3 {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 14px;

}

.head {

    display: flex;

    justify-content: space-between;

    align-items: end;

    gap: 10px;

    margin-bottom: 10px;

}

.title {

    font-weight: 800;

}

.rows {

    display: grid;

}

.row {

    display: flex;

    justify-content: space-between;

    gap: 12px;

    padding: 9px 0;

    border-bottom: 1px solid #ffffff11;

}

.row:last-child {

    border-bottom: 0;

}

.insight {

    border-left: 3px solid var(--blue);

    padding-left: 12px;

    line-height: 1.6;

    color: #c9d9e9;

}


/* =========================================================
   TRACKING MAP
   ========================================================= */

.tracking-shell {

    display: grid;

    grid-template-columns:
        minmax(0, 1.7fr)
        minmax(300px, .8fr);

    gap: 14px;

}

.tracking-map {

    position: relative;

    height: 500px;

    overflow: hidden;

    border:
        1px solid #1a3a55;

    border-radius: 12px;

    background:
        radial-gradient(
            circle at 55% 48%,
            #123451,
            #08121e 58%,
            #050c15
        );

}

.tracking-map svg {

    width: 100%;

    height: 100%;

    display: block;

}

.map-grid {

    stroke: #ffffff0b;

    stroke-width: 1;

}

.map-border {

    fill: none;

    stroke: #1c4b69;

    stroke-width: 1.2;

}

.route-line {

    fill: none;

    stroke: var(--cyan);

    stroke-width: 4;

    stroke-linecap: round;

    stroke-linejoin: round;

    filter:
        drop-shadow(
            0 0 8px
            #4de5ff88
        );

    stroke-dasharray: 900;

    stroke-dashoffset: 900;

    animation:
        routeDraw 2.4s
        ease forwards;

}

.prediction-line {

    fill: none;

    stroke: var(--purple);

    stroke-width: 3;

    stroke-dasharray: 10 10;

    animation:
        predictionDash 1.2s
        linear infinite;

    filter:
        drop-shadow(
            0 0 8px
            #a87cff99
        );

}

.trajectory-node {

    fill: #28a9ff;

    stroke: #bceaff;

    stroke-width: 2;

}

.trajectory-node.current {

    fill: var(--red);

    stroke: #ffd2d8;

    filter:
        drop-shadow(
            0 0 12px
            #ff5063
        );

}

.predicted-node {

    fill: var(--purple);

    stroke: #efe5ff;

    stroke-width: 2;

    animation:
        predictionPulse 1.5s
        ease-in-out infinite;

}

.event-radius {

    fill: #ff506310;

    stroke: #ff506366;

    stroke-width: 1;

    stroke-dasharray: 5 6;

    animation:
        radiusPulse 2s
        ease-in-out infinite;

}

.direction-arrow {

    stroke: var(--orange);

    stroke-width: 4;

    stroke-linecap: round;

    marker-end: url(#arrow);

    filter:
        drop-shadow(
            0 0 8px
            #ff9c42aa
        );

}

.map-label {

    fill: #cce5f9;

    font-size: 12px;

    font-weight: 600;

}

.map-label-muted {

    fill: #6e8da9;

    font-size: 10px;

}

.map-label-box {

    fill: #06101dcc;

    stroke: #294761;

    stroke-width: 1;

}

.tracking-side {

    display: grid;

    gap: 12px;

}

.live-card {

    position: relative;

    overflow: hidden;

}

.live-indicator {

    display: inline-flex;

    align-items: center;

    gap: 7px;

    color: var(--red);

    font-size: 10px;

    font-weight: 800;

    letter-spacing: .1em;

}

.live-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: var(--red);

    box-shadow:
        0 0 0 0
        #ff506377;

    animation:
        livePulse 1.5s infinite;

}

.location-big {

    font-size: 25px;

    font-weight: 900;

    margin: 12px 0 3px;

}

.prediction-card {

    border-color:
        #7b57bc88;

    background:
        linear-gradient(
            145deg,
            #15112a,
            #0a1220
        );

}

.prediction-title {

    display: flex;

    justify-content: space-between;

    align-items: center;

}

.prediction-badge {

    color: var(--purple);

    border:
        1px solid #8056c866;

    background:
        #7f50c811;

    border-radius: 20px;

    padding: 4px 8px;

    font-size: 9px;

}

.stat-grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 8px;

    margin-top: 12px;

}

.tracking-stat {

    border:
        1px solid #ffffff10;

    background:
        #07111d;

    border-radius: 8px;

    padding: 10px;

}

.tracking-stat .label {

    font-size: 9px;

}

.tracking-stat strong {

    display: block;

    margin-top: 5px;

    font-size: 15px;

}

.legend {

    display: flex;

    gap: 15px;

    flex-wrap: wrap;

    margin-top: 9px;

    font-size: 10px;

    color: var(--muted);

}

.legend-dot {

    display: inline-block;

    width: 9px;

    height: 9px;

    border-radius: 50%;

    margin-right: 5px;

}

.tracking-timeline {

    margin-top: 14px;

}

.timeline-scroll {

    display: flex;

    gap: 7px;

    overflow-x: auto;

    padding-bottom: 5px;

}

.timeline-event {

    min-width: 94px;

    padding: 10px;

    border:
        1px solid var(--line);

    background:
        #081321;

    border-radius: 9px;

}

.timeline-event.current {

    border-color:
        #ff506388;

    background:
        #ff50630b;

}

.timeline-time {

    color: var(--blue);

    font-size: 9px;

    font-weight: 800;

}

.timeline-wind {

    font-size: 16px;

    font-weight: 800;

    margin: 6px 0;

}

.wind-bar {

    height: 4px;

    border-radius: 10px;

    background:
        linear-gradient(
            90deg,
            #28a9ff,
            #ffd45a,
            #ff5063
        );

}

.direction-chip {

    margin-top: 6px;

    color: var(--orange);

    font-size: 9px;

}


/* =========================================================
   PIPELINE
   ========================================================= */

.pipeline-wrapper {

    display: grid;

    grid-template-columns:
        1.6fr 1fr;

    gap: 14px;

}

.pipeline-stage {

    position: relative;

    display: grid;

    grid-template-columns:
        42px 1fr auto;

    align-items: center;

    gap: 10px;

    padding: 13px;

    margin: 7px 0;

    border:
        1px solid var(--line);

    border-radius: 9px;

    background:
        #081321;

    transition:
        .35s ease;

}

.pipeline-stage.running {

    border-color:
        #238dcc;

    background:
        #0a2135;

    box-shadow:
        0 0 22px
        #28a9ff18;

}

.pipeline-stage.complete {

    border-color:
        #1c654d;

}

.pipeline-number {

    color: var(--blue);

    font-weight: 900;

}

.pipeline-status {

    font-size: 9px;

    padding: 4px 7px;

    border-radius: 20px;

    border:
        1px solid #21425d;

    color: var(--muted);

}

.pipeline-stage.complete
.pipeline-status {

    color: var(--green);

    border-color:
        #1c654d;

}

.progress-wrap {

    height: 8px;

    background:
        #081321;

    border-radius: 20px;

    overflow: hidden;

    margin-top: 12px;

}

.progress-bar {

    width: 0%;

    height: 100%;

    background:
        linear-gradient(
            90deg,
            var(--blue),
            var(--cyan),
            var(--purple)
        );

    transition:
        width .45s ease;

}


/* =========================================================
   WEATHER
   ========================================================= */

.heat {

    height: 250px;

    border-radius: 9px;

    background:
        conic-gradient(
            from 110deg,
            #173e76,
            #1ab4e6,
            #ffd74e,
            #ff8b35,
            #e82b49,
            #173e76
        );

    position: relative;

    overflow: hidden;

}

.heat:after {

    content: "";

    position: absolute;

    inset: 0;

    background:
        linear-gradient(
            90deg,
            #06101c99,
            #fff0,
            #06101c66
        );

}


/* =========================================================
   DOWNscaling
   ========================================================= */

.down-grid {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 14px;

}


/* =========================================================
   RISK
   ========================================================= */

.risk-meter {

    height: 250px;

    display: grid;

    place-items: center;

    position: relative;

}

.risk-ring {

    width: 185px;

    height: 185px;

    border-radius: 50%;

    background:
        conic-gradient(
            var(--red) 0 89.87%,
            #26364b 89.87%
        );

    display: grid;

    place-items: center;

    box-shadow:
        0 0 45px
        #ff506326;

}

.risk-ring:after {

    content: "";

    position: absolute;

    width: 140px;

    height: 140px;

    border-radius: 50%;

    background:
        #0a1422;

}

.risk-number {

    position: relative;

    z-index: 2;

    text-align: center;

}

.risk-number b {

    font-size: 34px;

}


/* =========================================================
   LOCATION
   ========================================================= */

.location-search {

    display: grid;

    grid-template-columns:
        1fr auto;

    gap: 8px;

}

.location-input {

    width: 100%;

    padding: 11px;

    border-radius: 8px;

    border:
        1px solid var(--line);

    background:
        #06101b;

    color: white;

    outline: none;

}

.location-input:focus {

    border-color:
        var(--blue);

}

.coordinate-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr auto;

    gap: 8px;

}

.location-result {

    min-height: 130px;

}

.location-risk {

    border:
        1px solid var(--line);

    border-radius: 10px;

    padding: 15px;

}

.location-risk.extreme {

    border-color:
        #ff506388;

    box-shadow:
        0 0 25px
        #ff506315;

}

.location-risk.high {

    border-color:
        #ff9c4288;

}

.location-risk.moderate {

    border-color:
        #ffd45a88;

}

.location-risk.low {

    border-color:
        #39df9288;

}


/* =========================================================
   GRAPH
   ========================================================= */

.graph-box {

    height: 370px;

    position: relative;

    background:
        #07111d;

    border:
        1px solid var(--line);

    border-radius: 10px;

    overflow: hidden;

}

.graph-node {

    position: absolute;

    width: 13px;

    height: 13px;

    border-radius: 50%;

    background:
        var(--blue);

    box-shadow:
        0 0 14px
        var(--blue);

}

.graph-node.current {

    background:
        var(--red);

    box-shadow:
        0 0 20px
        var(--red);

}


/* =========================================================
   TABLE / CODE
   ========================================================= */

.code {

    background:
        #040a11;

    border:
        1px solid var(--line);

    border-radius: 9px;

    padding: 14px;

    white-space: pre-wrap;

    color:
        #bfe2ff;

    overflow: auto;

}

.chips {

    display: flex;

    gap: 7px;

    flex-wrap: wrap;

}

.chip {

    border:
        1px solid var(--line);

    background:
        #091727;

    border-radius: 7px;

    padding: 7px 9px;

    font-size: 11px;

    color:
        #bed0e3;

}


/* =========================================================
   MODAL
   ========================================================= */

.modal {

    position: fixed;

    inset: 0;

    background:
        #000b;

    z-index: 50;

    display: grid;

    place-items: center;

}

.modal.hidden {

    display: none;

}

.modal-box {

    width:
        min(600px, 92vw);

    background:
        #0d1828;

    border:
        1px solid var(--line);

    border-radius: 13px;

    padding: 24px;

}


/* =========================================================
   TOAST
   ========================================================= */

.toast {

    position: fixed;

    right: 20px;

    bottom: 20px;

    background:
        #112a42;

    border:
        1px solid #28658c;

    padding: 10px 13px;

    border-radius: 8px;

    z-index: 60;

}


/* =========================================================
   PRESENTATION MODE
   ========================================================= */

.presentation .sidebar {

    display: none;

}

.presentation .main {

    margin-left: 0;

    width: 100%;

    padding: 35px;

}

.presentation .tracking-map {

    height: 620px;

}


/* =========================================================
   ANIMATIONS
   ========================================================= */

@keyframes routeDraw {

    to {
        stroke-dashoffset: 0;
    }

}

@keyframes predictionDash {

    to {
        stroke-dashoffset: -40;
    }

}

@keyframes predictionPulse {

    0%,
    100% {
        opacity: .65;
        r: 8;
    }

    50% {
        opacity: 1;
        r: 12;
    }

}

@keyframes radiusPulse {

    0%,
    100% {
        opacity: .45;
    }

    50% {
        opacity: .9;
    }

}

@keyframes livePulse {

    0% {
        box-shadow:
            0 0 0 0
            #ff506377;
    }

    70% {
        box-shadow:
            0 0 0 9px
            #ff506300;
    }

    100% {
        box-shadow:
            0 0 0 0
            #ff506300;
    }

}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media(max-width:1100px) {

    .cards {

        grid-template-columns:
            repeat(3, 1fr);

    }

    .grid2,
    .grid3,
    .tracking-shell,
    .pipeline-wrapper {

        grid-template-columns:
            1fr;

    }

}

@media(max-width:760px) {

    .sidebar {

        position: relative;

        width: 100%;

        height: auto;

        border-right: 0;

        border-bottom:
            1px solid var(--line);

    }

    .app {

        display: block;

    }

    .main {

        margin-left: 0;

        width: 100%;

        padding: 18px;

    }

    .nav {

        display: flex;

        overflow-x: auto;

    }

    .nav button {

        white-space: nowrap;

    }

    .topbar {

        flex-direction: column;

        align-items: flex-start;

    }

    .cards {

        grid-template-columns:
            repeat(2, 1fr);

    }

    .coordinate-grid {

        grid-template-columns:
            1fr;

    }

    .location-search {

        grid-template-columns:
            1fr;

    }

}

</style>

</head>


<body>


<div class="app">


<aside class="sidebar">

    <div class="brand">

        EXTREMEWEATHER

        <span>AI</span>

    </div>

    <div class="system-online">

        ● SYSTEM ONLINE

    </div>

    <nav
        id="navigation"
        class="nav"
    ></nav>

</aside>


<main class="main">


<div class="topbar">

    <div>

        <div class="eyebrow">

            SIH 2026 · PROBLEM 26078 · NCMRWF

        </div>

        <h1 id="pageTitle">

            Extreme Weather Intelligence

        </h1>

        <div
            id="pageSubtitle"
            class="subtitle"
        >

            AI-powered detection,
            tracking and risk assessment
            of extreme weather anomalies.

        </div>

    </div>


    <div class="actions">

        <span class="badge">

            PROTOTYPE MODE

        </span>

        <button
            class="btn"
            onclick="showInfo()"
        >

            ⓘ DEMO INFO

        </button>

        <button
            class="btn primary"
            onclick="togglePresentation()"
        >

            PRESENTATION MODE

        </button>

    </div>

</div>


<div id="content"></div>


</main>

</div>


<div
    id="modal"
    class="modal hidden"
>

    <div class="modal-box">

        <h2>Prototype Mode</h2>

        <p class="subtitle">

            This dashboard is a proof-of-concept.

            It currently uses synthetic weather
            data and prototype AI components.

            Real operational deployment would
            require validation against historical
            observations and operational NWP
            datasets.

            It is not connected to live
            NCUM / NEPS-G feeds.

        </p>

        <button
            class="btn primary"
            onclick="closeInfo()"
        >

            Close

        </button>

    </div>

</div>


<script>


/* ============================================================
   DATA FROM FLASK
   ============================================================ */

const ALERT =
    {{ alert | tojson }};

const TRAJECTORY =
    {{ trajectory | tojson }};

const GNN =
    {{ gnn | tojson }};

const DOWNSCALING =
    {{ downscaling | tojson }};


/* ============================================================
   NAVIGATION
   ============================================================ */

const NAV_ITEMS = [

    ["dashboard", "Dashboard"],

    ["weather", "Weather Monitor"],

    ["tracking", "AI Tracking"],

    ["location", "My Location"],

    ["downscaling", "Downscaling"],

    ["risk", "Risk & Alerts"],

    ["pipeline", "AI Pipeline"],

    ["api", "API"],

    ["about", "About"]

];


const PAGE_INFO = {

    dashboard: [
        "Extreme Weather Intelligence",
        "AI-powered detection, tracking and risk assessment of extreme weather anomalies."
    ],

    weather: [
        "Weather Monitor",
        "Prototype anomaly detection and weather-field analysis."
    ],

    tracking: [
        "AI Tracking",
        "Spatio-temporal graph learning and trajectory prediction."
    ],

    location: [
        "My Location",
        "Check prototype weather-event exposure around your location."
    ],

    downscaling: [
        "Extreme-Aware Downscaling",
        "Prototype demonstration of extreme-preserving spatial enhancement."
    ],

    risk: [
        "Risk & Alerts",
        "Localized risk assessment and emergency alert generation."
    ],

    pipeline: [
        "AI Pipeline",
        "End-to-end processing and visual pipeline execution."
    ],

    api: [
        "REST API",
        "Machine-readable prototype alert and trajectory interfaces."
    ],

    about: [
        "About ExtremeWeather AI",
        "Project, data sources, technology and prototype scope."
    ]

};


function buildNavigation() {

    const nav =
        document.getElementById(
            "navigation"
        );

    nav.innerHTML = "";

    NAV_ITEMS.forEach(
        ([id, label]) => {

            const button =
                document.createElement(
                    "button"
                );

            button.textContent =
                label;

            button.dataset.page =
                id;

            button.onclick =
                () => showPage(id);

            nav.appendChild(button);

        }
    );

}


/* ============================================================
   PAGE SWITCHING
   ============================================================ */

function showPage(page) {

    document
        .querySelectorAll(
            "#navigation button"
        )
        .forEach(button => {

            button.classList.toggle(
                "active",
                button.dataset.page === page
            );

        });


    document.getElementById(
        "pageTitle"
    ).textContent =
        PAGE_INFO[page][0];


    document.getElementById(
        "pageSubtitle"
    ).textContent =
        PAGE_INFO[page][1];


    const functions = {

        dashboard: renderDashboard,

        weather: renderWeather,

        tracking: renderTracking,

        location: renderLocation,

        downscaling: renderDownscaling,

        risk: renderRisk,

        pipeline: renderPipeline,

        api: renderAPI,

        about: renderAbout

    };


    if (functions[page]) {

        functions[page]();

    }

}


/* ============================================================
   DASHBOARD
   ============================================================ */

function renderDashboard() {

    const content =
        document.getElementById(
            "content"
        );


    content.innerHTML = `

        <div class="cards">

            <div class="card">

                <div class="label">
                    Risk Level
                </div>

                <div class="value red">
                    EXTREME
                </div>

                <div class="small">
                    Current anomaly classification
                </div>

            </div>


            <div class="card">

                <div class="label">
                    Maximum Wind
                </div>

                <div class="value red">
                    ${ALERT.maximum_wind_ms}
                    m/s
                </div>

                <div class="small">
                    Detected extreme value
                </div>

            </div>


            <div class="card">

                <div class="label">
                    Intensity Score
                </div>

                <div class="value yellow">
                    ${ALERT.intensity_score}
                    / 100
                </div>

                <div class="small">
                    Prototype AI risk score
                </div>

            </div>


            <div class="card">

                <div class="label">
                    Movement
                </div>

                <div class="value blue">
                    ${ALERT.movement_km}
                    km
                </div>

                <div class="small">
                    Observed displacement
                </div>

            </div>


            <div class="card">

                <div class="label">
                    Direction
                </div>

                <div class="value blue">
                    NORTH EAST
                </div>

                <div class="small">
                    Current movement
                </div>

            </div>


            <div class="card">

                <div class="label">
                    Alert Radius
                </div>

                <div class="value orange">
                    ${ALERT.affected_radius_km}
                    km
                </div>

                <div class="small">
                    Localized alert region
                </div>

            </div>

        </div>


        <div class="section tracking-shell">

            <div class="card">

                <div class="head">

                    <div>

                        <div class="title">
                            LIVE ANOMALY TRACKING
                        </div>

                        <div class="small">
                            Current anomaly trajectory
                        </div>

                    </div>

                    <span class="badge">
                        DEMO DATA
                    </span>

                </div>

                ${trackingMapHTML()}

            </div>


            <div class="tracking-side">

                <div class="card live-card">

                    <div class="live-indicator">

                        <span class="live-dot"></span>

                        LIVE EVENT

                    </div>

                    <div class="location-big">

                        ${formatCoord(
                            ALERT.latitude,
                            ALERT.longitude
                        )}

                    </div>

                    <div class="small">
                        Current anomaly position
                    </div>

                    <div class="stat-grid">

                        <div class="tracking-stat">

                            <div class="label">
                                WIND
                            </div>

                            <strong class="red">
                                ${ALERT.maximum_wind_ms}
                                m/s
                            </strong>

                        </div>

                        <div class="tracking-stat">

                            <div class="label">
                                INTENSITY
                            </div>

                            <strong class="yellow">
                                ${ALERT.intensity_score}
                            </strong>

                        </div>

                    </div>

                </div>


                <div class="card prediction-card">

                    <div class="prediction-title">

                        <div class="title">
                            AI PREDICTION
                        </div>

                        <span class="prediction-badge">
                            NEXT STATE
                        </span>

                    </div>

                    <div class="location-big purple">

                        ${formatCoord(
                            GNN.predicted_latitude,
                            GNN.predicted_longitude
                        )}

                    </div>

                    <div class="small">
                        Predicted next anomaly location
                    </div>

                    <div class="stat-grid">

                        <div class="tracking-stat">

                            <div class="label">
                                MODEL
                            </div>

                            <strong>
                                ST-GNN
                            </strong>

                        </div>

                        <div class="tracking-stat">

                            <div class="label">
                                DIRECTION
                            </div>

                            <strong class="orange">
                                NE
                            </strong>

                        </div>

                    </div>

                </div>

            </div>

        </div>


        <div class="section card">

            <div class="title">
                TRAJECTORY HISTORY
            </div>

            <div class="small">
                Observed anomaly states across forecast time
            </div>

            <div
                id="dashboardTimeline"
                class="tracking-timeline"
            ></div>

        </div>

    `;


    renderTimeline(
        document.getElementById(
            "dashboardTimeline"
        )
    );

}


/* ============================================================
   TRACKING MAP
   ============================================================ */

function trackingMapHTML() {

    return `

        <div
            id="trackingMap"
            class="tracking-map"
        >

            <svg
                id="trackingSVG"
                viewBox="0 0 1000 520"
                preserveAspectRatio="none"
            >

                <defs>

                    <marker
                        id="arrow"
                        markerWidth="8"
                        markerHeight="8"
                        refX="7"
                        refY="4"
                        orient="auto"
                    >

                        <path
                            d="M0,0 L8,4 L0,8 Z"
                            fill="#ff9c42"
                        />

                    </marker>

                </defs>


                <rect
                    x="0"
                    y="0"
                    width="1000"
                    height="520"
                    fill="#06101b"
                />


                <g id="mapGrid"></g>


                <rect
                    class="map-border"
                    x="20"
                    y="20"
                    width="960"
                    height="480"
                    rx="16"
                />


                <circle
                    id="eventRadius"
                    class="event-radius"
                    cx="0"
                    cy="0"
                    r="55"
                />


                <polyline
                    id="routeLine"
                    class="route-line"
                    points=""
                />


                <line
                    id="predictionLine"
                    class="prediction-line"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="0"
                />


                <line
                    id="directionArrow"
                    class="direction-arrow"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="0"
                />


                <g id="trajectoryNodes"></g>


                <circle
                    id="predictedNode"
                    class="predicted-node"
                    cx="0"
                    cy="0"
                    r="9"
                />


                <text
                    x="35"
                    y="48"
                    class="map-label"
                >
                    BAY OF BENGAL
                </text>


                <text
                    x="35"
                    y="66"
                    class="map-label-muted"
                >
                    SPATIO-TEMPORAL WEATHER FIELD
                </text>


                <g id="currentLabel">

                    <rect
                        class="map-label-box"
                        x="0"
                        y="0"
                        width="165"
                        height="44"
                        rx="6"
                    />

                    <text
                        id="currentLabelText"
                        x="0"
                        y="0"
                        class="map-label"
                    ></text>

                </g>


                <g id="predictionLabel">

                    <rect
                        class="map-label-box"
                        x="0"
                        y="0"
                        width="175"
                        height="44"
                        rx="6"
                    />

                    <text
                        id="predictionLabelText"
                        x="0"
                        y="0"
                        class="map-label"
                    ></text>

                </g>

            </svg>

        </div>


        <div class="legend">

            <span>
                <i
                    class="legend-dot"
                    style="background:#28a9ff"
                ></i>
                Historical path
            </span>

            <span>
                <i
                    class="legend-dot"
                    style="background:#ff5063"
                ></i>
                Current anomaly
            </span>

            <span>
                <i
                    class="legend-dot"
                    style="background:#a87cff"
                ></i>
                GNN prediction
            </span>

            <span>
                <i
                    class="legend-dot"
                    style="background:#ff9c42"
                ></i>
                Movement direction
            </span>

        </div>

    `;
}


function createMapGrid() {

    const grid =
        document.getElementById(
            "mapGrid"
        );

    if (!grid) {
        return;
    }

    grid.innerHTML = "";

    for (
        let x = 100;
        x < 1000;
        x += 100
    ) {

        grid.innerHTML += `
            <line
                class="map-grid"
                x1="${x}"
                y1="20"
                x2="${x}"
                y2="500"
            />
        `;

    }

    for (
        let y = 100;
        y < 520;
        y += 80
    ) {

        grid.innerHTML += `
            <line
                class="map-grid"
                x1="20"
                y1="${y}"
                x2="980"
                y2="${y}"
            />
        `;

    }

}


function calculateMapPoint(
    latitude,
    longitude,
    bounds
) {

    const x =
        70
        +
        (
            (longitude - bounds.minLon)
            /
            (bounds.maxLon - bounds.minLon)
        )
        *
        860;

    const y =
        460
        -
        (
            (latitude - bounds.minLat)
            /
            (bounds.maxLat - bounds.minLat)
        )
        *
        400;

    return {
        x,
        y
    };

}


function renderTrackingMap() {

    const svg =
        document.getElementById(
            "trackingSVG"
        );

    if (!svg) {
        return;
    }

    createMapGrid();


    const allPoints =
        TRAJECTORY.map(
            point => ({
                lat:
                    Number(
                        point.latitude
                    ),

                lon:
                    Number(
                        point.longitude
                    )
            })
        );


    allPoints.push({
        lat:
            Number(
                GNN.predicted_latitude
            ),

        lon:
            Number(
                GNN.predicted_longitude
            )
    });


    let minLat =
        Math.min(
            ...allPoints.map(
                point => point.lat
            )
        );

    let maxLat =
        Math.max(
            ...allPoints.map(
                point => point.lat
            )
        );

    let minLon =
        Math.min(
            ...allPoints.map(
                point => point.lon
            )
        );

    let maxLon =
        Math.max(
            ...allPoints.map(
                point => point.lon
            )
        );


    const latPadding =
        Math.max(
            (maxLat - minLat) * 0.25,
            1
        );

    const lonPadding =
        Math.max(
            (maxLon - minLon) * 0.25,
            1
        );


    const bounds = {

        minLat:
            minLat - latPadding,

        maxLat:
            maxLat + latPadding,

        minLon:
            minLon - lonPadding,

        maxLon:
            maxLon + lonPadding

    };


    const points =
        TRAJECTORY.map(
            point =>
                calculateMapPoint(
                    Number(
                        point.latitude
                    ),
                    Number(
                        point.longitude
                    ),
                    bounds
                )
        );


    const current =
        points[
            points.length - 1
        ];


    const predicted =
        calculateMapPoint(
            Number(
                GNN.predicted_latitude
            ),
            Number(
                GNN.predicted_longitude
            ),
            bounds
        );


    const route =
        document.getElementById(
            "routeLine"
        );

    route.setAttribute(
        "points",
        points
            .map(
                point =>
                    `${point.x},${point.y}`
            )
            .join(" ")
    );


    const predictionLine =
        document.getElementById(
            "predictionLine"
        );

    predictionLine.setAttribute(
        "x1",
        current.x
    );

    predictionLine.setAttribute(
        "y1",
        current.y
    );

    predictionLine.setAttribute(
        "x2",
        predicted.x
    );

    predictionLine.setAttribute(
        "y2",
        predicted.y
    );


    const previous =
        points.length >= 2
            ? points[
                points.length - 2
            ]
            : current;


    const directionArrow =
        document.getElementById(
            "directionArrow"
        );


    const dx =
        current.x - previous.x;

    const dy =
        current.y - previous.y;

    const arrowLength =
        55;


    const magnitude =
        Math.sqrt(
            dx * dx +
            dy * dy
        ) || 1;


    const unitX =
        dx / magnitude;

    const unitY =
        dy / magnitude;


    directionArrow.setAttribute(
        "x1",
        current.x - unitX * 10
    );

    directionArrow.setAttribute(
        "y1",
        current.y - unitY * 10
    );

    directionArrow.setAttribute(
        "x2",
        current.x + unitX * arrowLength
    );

    directionArrow.setAttribute(
        "y2",
        current.y + unitY * arrowLength
    );


    const nodes =
        document.getElementById(
            "trajectoryNodes"
        );

    nodes.innerHTML = "";


    points.forEach(
        (point, index) => {

            const circle =
                document.createElementNS(
                    "http://www.w3.org/2000/svg",
                    "circle"
                );

            circle.setAttribute(
                "cx",
                point.x
            );

            circle.setAttribute(
                "cy",
                point.y
            );

            circle.setAttribute(
                "r",
                index === points.length - 1
                    ? 10
                    : 5
            );

            circle.setAttribute(
                "class",
                index === points.length - 1
                    ? "trajectory-node current"
                    : "trajectory-node"
            );


            circle.style.animationDelay =
                `${index * 0.05}s`;


            nodes.appendChild(
                circle
            );

        }
    );


    const predictedNode =
        document.getElementById(
            "predictedNode"
        );


    predictedNode.setAttribute(
        "cx",
        predicted.x
    );

    predictedNode.setAttribute(
        "cy",
        predicted.y
    );


    const radius =
        document.getElementById(
            "eventRadius"
        );

    radius.setAttribute(
        "cx",
        current.x
    );

    radius.setAttribute(
        "cy",
        current.y
    );


    const currentLabel =
        document.getElementById(
            "currentLabel"
        );

    currentLabel.setAttribute(
        "transform",
        `
            translate(
                ${current.x - 82},
                ${current.y + 18}
            )
        `
    );


    document.getElementById(
        "currentLabelText"
    ).setAttribute(
        "x",
        10
    );

    document.getElementById(
        "currentLabelText"
    ).setAttribute(
        "y",
        19
    );

    document.getElementById(
        "currentLabelText"
    ).textContent =
        "CURRENT · EXTREME";


    const predictionLabel =
        document.getElementById(
            "predictionLabel"
        );


    predictionLabel.setAttribute(
        "transform",
        `
            translate(
                ${predicted.x - 87},
                ${predicted.y - 55}
            )
        `
    );


    document.getElementById(
        "predictionLabelText"
    ).setAttribute(
        "x",
        10
    );

    document.getElementById(
        "predictionLabelText"
    ).setAttribute(
        "y",
        19
    );

    document.getElementById(
        "predictionLabelText"
    ).textContent =
        "GNN · PREDICTED";

}


/* ============================================================
   AI TRACKING PAGE
   ============================================================ */

function renderTracking() {

    const content =
        document.getElementById(
            "content"
        );


    content.innerHTML = `

        <div class="tracking-shell">


            <div class="card">

                <div class="head">

                    <div>

                        <div class="live-indicator">

                            <span class="live-dot"></span>

                            LIVE EVENT TRACKING

                        </div>

                        <div class="title"
                             style="margin-top:7px">

                            Spatio-Temporal AI Trajectory

                        </div>

                        <div class="small">

                            Historical anomaly path →
                            current state →
                            predicted next state

                        </div>

                    </div>


                    <span class="badge">

                        NEXT TIMESTEP

                    </span>

                </div>


                ${trackingMapHTML()}

            </div>


            <div class="tracking-side">


                <div class="card live-card">

                    <div class="title">

                        CURRENT ANOMALY

                    </div>


                    <div class="location-big red">

                        ${formatCoord(
                            ALERT.latitude,
                            ALERT.longitude
                        )}

                    </div>


                    <div class="small">

                        Latest detected state

                    </div>


                    <div class="stat-grid">

                        <div class="tracking-stat">

                            <div class="label">
                                WIND
                            </div>

                            <strong class="red">
                                ${ALERT.maximum_wind_ms}
                                m/s
                            </strong>

                        </div>


                        <div class="tracking-stat">

                            <div class="label">
                                INTENSITY
                            </div>

                            <strong class="yellow">
                                ${ALERT.intensity_score}
                            </strong>

                        </div>


                        <div class="tracking-stat">

                            <div class="label">
                                MOVEMENT
                            </div>

                            <strong class="blue">
                                ${ALERT.movement_km}
                                km
                            </strong>

                        </div>


                        <div class="tracking-stat">

                            <div class="label">
                                DIRECTION
                            </div>

                            <strong class="orange">
                                ${ALERT.direction}
                            </strong>

                        </div>

                    </div>

                </div>


                <div class="card prediction-card">

                    <div class="prediction-title">

                        <div class="title">

                            AI PREDICTION

                        </div>

                        <span class="prediction-badge">

                            GNN

                        </span>

                    </div>


                    <div class="location-big purple">

                        ${formatCoord(
                            GNN.predicted_latitude,
                            GNN.predicted_longitude
                        )}

                    </div>


                    <div class="small">

                        Estimated next anomaly location

                    </div>


                    <div class="stat-grid">

                        <div class="tracking-stat">

                            <div class="label">
                                CURRENT LAT
                            </div>

                            <strong>
                                ${Number(
                                    GNN.current_latitude
                                ).toFixed(2)}
                                °
                            </strong>

                        </div>


                        <div class="tracking-stat">

                            <div class="label">
                                CURRENT LON
                            </div>

                            <strong>
                                ${Number(
                                    GNN.current_longitude
                                ).toFixed(2)}
                                °
                            </strong>

                        </div>


                        <div class="tracking-stat">

                            <div class="label">
                                PREDICTED LAT
                            </div>

                            <strong class="purple">
                                ${Number(
                                    GNN.predicted_latitude
                                ).toFixed(2)}
                                °
                            </strong>

                        </div>


                        <div class="tracking-stat">

                            <div class="label">
                                PREDICTED LON
                            </div>

                            <strong class="purple">
                                ${Number(
                                    GNN.predicted_longitude
                                ).toFixed(2)}
                                °
                            </strong>

                        </div>

                    </div>

                </div>


                <div class="card">

                    <div class="title">

                        MODEL STATUS

                    </div>


                    <div class="row">

                        <span>
                            Model
                        </span>

                        <b>
                            Spatio-Temporal GNN
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Graph
                        </span>

                        <b class="green">
                            10 Nodes
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Temporal Links
                        </span>

                        <b class="green">
                            9
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Spatial Links
                        </span>

                        <b class="green">
                            15
                        </b>

                    </div>


                    <p class="small">

                        Prototype prediction only.
                        The current model is not a
                        production spherical /
                        icosahedral GNN.

                    </p>

                </div>


            </div>

        </div>


        <div class="section card">

            <div class="head">

                <div>

                    <div class="title">

                        TRAJECTORY TIMELINE

                    </div>

                    <div class="small">

                        Wind intensity and anomaly
                        movement through forecast time

                    </div>

                </div>

                <span class="badge">

                    ${TRAJECTORY.length} OBSERVATIONS

                </span>

            </div>


            <div
                id="trackingTimeline"
                class="timeline-scroll"
            ></div>

        </div>


        <div class="section grid3">


            <div class="card">

                <div class="label">
                    CURRENT STATE
                </div>

                <div class="value red">

                    ${formatCoord(
                        ALERT.latitude,
                        ALERT.longitude
                    )}

                </div>

                <div class="small">
                    Latest detected anomaly
                </div>

            </div>


            <div class="card">

                <div class="label">
                    PREDICTED STATE
                </div>

                <div class="value purple">

                    ${formatCoord(
                        GNN.predicted_latitude,
                        GNN.predicted_longitude
                    )}

                </div>

                <div class="small">
                    GNN next-state estimate
                </div>

            </div>


            <div class="card">

                <div class="label">
                    TRAJECTORY TREND
                </div>

                <div class="value orange">
                    ↗ NORTH EAST
                </div>

                <div class="small">
                    Based on recent movement
                </div>

            </div>


        </div>

    `;


    renderTrackingMap();


    renderTimeline(
        document.getElementById(
            "trackingTimeline"
        )
    );

}


/* ============================================================
   TIMELINE
   ============================================================ */

function renderTimeline(container) {

    if (!container) {
        return;
    }


    const maximum =
        Math.max(
            ...TRAJECTORY.map(
                point =>
                    Number(
                        point.max_wind
                    )
            )
        );


    container.innerHTML =
        TRAJECTORY.map(
            (point, index) => {

                const wind =
                    Number(
                        point.max_wind
                    );


                const width =
                    Math.max(
                        5,
                        (wind / maximum) * 100
                    );


                return `

                    <div
                        class="
                            timeline-event
                            ${
                                index ===
                                TRAJECTORY.length - 1
                                    ? "current"
                                    : ""
                            }
                        "
                    >

                        <div class="timeline-time">

                            ${point.time}

                        </div>


                        <div class="timeline-wind">

                            ${wind.toFixed(2)}

                            <span
                                class="small"
                            >
                                m/s
                            </span>

                        </div>


                        <div class="wind-bar"
                             style="
                                width:${width}%;
                             "
                        ></div>


                        <div class="small"
                             style="margin-top:7px">

                            ${Number(
                                point.latitude
                            ).toFixed(2)}
                            °N

                            <br>

                            ${Number(
                                point.longitude
                            ).toFixed(2)}
                            °E

                        </div>


                        <div class="direction-chip">

                            ↗
                            ${point.direction || "NE"}

                        </div>

                    </div>

                `;

            }
        ).join("");

}


/* ============================================================
   WEATHER PAGE
   ============================================================ */

function renderWeather() {

    document.getElementById(
        "content"
    ).innerHTML = `

        <div class="grid2">


            <div class="card">

                <div class="title">
                    INPUT WEATHER FIELD
                </div>

                <div class="small">
                    Prototype synthetic weather field
                </div>

                <div class="heat section"></div>


                <div class="legend">

                    <span>
                        Normal
                    </span>

                    <span>
                        Moderate
                    </span>

                    <span>
                        Severe
                    </span>

                    <span>
                        Extreme
                    </span>

                </div>

            </div>


            <div class="card">

                <div class="title">
                    EXTREME ANOMALY DETECTION
                </div>


                <div class="rows">

                    <div class="row">

                        <span>
                            Maximum detected wind
                        </span>

                        <b class="red">
                            62.91 m/s
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Prototype threshold
                        </span>

                        <b>
                            40 m/s
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Anomaly status
                        </span>

                        <b class="red">
                            DETECTED
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Region
                        </span>

                        <b>
                            Dynamic
                        </b>

                    </div>

                </div>


                <div class="section">

                    <div class="title">
                        CURRENT REGION
                    </div>

                    <div class="row">

                        <span>
                            Latitude
                        </span>

                        <b>
                            18.0909° N
                        </b>

                    </div>

                    <div class="row">

                        <span>
                            Longitude
                        </span>

                        <b>
                            89.3636° E
                        </b>

                    </div>

                    <div class="row">

                        <span>
                            Classification
                        </span>

                        <b class="red">
                            EXTREME
                        </b>

                    </div>

                </div>

            </div>

        </div>


        <div class="section grid3">


            <div class="card">

                <div class="title">
                    DETECTION
                </div>

                <div class="value red">
                    ACTIVE
                </div>

                <div class="small">
                    Extreme weather region identified
                </div>

            </div>


            <div class="card">

                <div class="title">
                    TRACKING
                </div>

                <div class="value blue">
                    NE
                </div>

                <div class="small">
                    Current movement direction
                </div>

            </div>


            <div class="card">

                <div class="title">
                    SEVERITY
                </div>

                <div class="value red">
                    EXTREME
                </div>

                <div class="small">
                    Prototype classification
                </div>

            </div>

        </div>

    `;

}


/* ============================================================
   LOCATION PAGE
   ============================================================ */

function renderLocation() {

    document.getElementById(
        "content"
    ).innerHTML = `

        <div class="grid2">


            <div class="card">

                <div class="title">
                    CHECK YOUR LOCATION
                </div>

                <div class="small section">
                    Detect your browser location
                    or search for a city.
                </div>


                <button
                    id="detectLocationButton"
                    class="btn primary section"
                    onclick="getMyLocation()"
                >

                    📍 Detect My Location

                </button>


                <div
                    id="gpsStatus"
                    class="small section"
                >

                    Location permission is
                    required for automatic detection.

                </div>


                <div class="title section">
                    SEARCH CITY
                </div>


                <div class="location-search section">

                    <input
                        id="cityInput"
                        class="location-input"
                        placeholder="Example: Bengaluru"
                    >

                    <button
                        class="btn"
                        onclick="searchCity()"
                    >

                        Search

                    </button>

                </div>


                <div class="title section">
                    ENTER COORDINATES
                </div>


                <div class="coordinate-grid section">

                    <input
                        id="manualLat"
                        class="location-input"
                        placeholder="Latitude"
                    >

                    <input
                        id="manualLon"
                        class="location-input"
                        placeholder="Longitude"
                    >

                    <button
                        class="btn"
                        onclick="checkManualLocation()"
                    >

                        Check

                    </button>

                </div>


                <button
                    class="btn section"
                    onclick="clearLocation()"
                >

                    Clear Saved Location

                </button>

            </div>


            <div class="card">

                <div class="title">
                    LOCATION RISK
                </div>

                <div
                    id="locationResult"
                    class="location-result section"
                >

                    <div class="small">

                        No location selected yet.

                    </div>

                </div>

            </div>


        </div>


        <div class="section grid3">


            <div class="card">

                <div class="label">
                    CURRENT EVENT
                </div>

                <div class="value red">
                    18.09° N
                </div>

                <div class="small">
                    89.36° E
                </div>

            </div>


            <div class="card">

                <div class="label">
                    PREDICTED EVENT
                </div>

                <div class="value purple">
                    18.97° N
                </div>

                <div class="small">
                    90.10° E
                </div>

            </div>


            <div class="card">

                <div class="label">
                    ALERT RADIUS
                </div>

                <div class="value orange">
                    5 km
                </div>

                <div class="small">
                    Prototype affected region
                </div>

            </div>

        </div>

    `;


    loadSavedLocation();

}


function getMyLocation() {

    const result =
        document.getElementById(
            "locationResult"
        );

    const status =
        document.getElementById(
            "gpsStatus"
        );

    const button =
        document.getElementById(
            "detectLocationButton"
        );


    if (
        !navigator.geolocation
    ) {

        status.innerHTML = `
            <span class="red">
                ❌ Geolocation is not
                supported by this browser.
            </span>
        `;

        return;

    }


    button.disabled = true;

    button.textContent =
        "📍 Detecting...";


    status.innerHTML =
        "Requesting your browser location permission...";


    navigator.geolocation.getCurrentPosition(

        function(position) {

            const latitude =
                position.coords.latitude;

            const longitude =
                position.coords.longitude;

            const accuracy =
                position.coords.accuracy;


            localStorage.setItem(
                "userLatitude",
                latitude
            );

            localStorage.setItem(
                "userLongitude",
                longitude
            );


            document.getElementById(
                "manualLat"
            ).value =
                latitude.toFixed(6);


            document.getElementById(
                "manualLon"
            ).value =
                longitude.toFixed(6);


            status.innerHTML = `

                <span class="green">

                    ✓ Location detected

                </span>

                <br>

                Accuracy:
                ${Math.round(accuracy)}
                metres

            `;


            calculateLocationRisk(
                latitude,
                longitude
            );


            button.disabled = false;

            button.textContent =
                "📍 Detect My Location";

        },


        function(error) {

            console.error(
                "GPS ERROR:",
                error
            );


            let message =
                "Unable to detect your location.";


            if (
                error.code ===
                error.PERMISSION_DENIED
            ) {

                message =
                    "Location permission was denied.";

            }

            else if (
                error.code ===
                error.POSITION_UNAVAILABLE
            ) {

                message =
                    "Your location is currently unavailable.";

            }

            else if (
                error.code ===
                error.TIMEOUT
            ) {

                message =
                    "Location detection timed out.";

            }


            status.innerHTML = `

                <span class="red">

                    ❌ ${message}

                </span>

            `;


            result.innerHTML = `

                <div class="card">

                    <div class="red">

                        ${message}

                    </div>

                    <br>

                    <div class="small">

                        If Chrome blocked location,
                        click the 🔒 icon beside
                        the address bar and set

                        <strong>
                            Location → Allow
                        </strong>.

                        <br><br>

                        Then reload the page and
                        try again.

                    </div>

                </div>

            `;


            button.disabled = false;

            button.textContent =
                "📍 Detect My Location";

        },


        {

            enableHighAccuracy: true,

            timeout: 30000,

            maximumAge: 0

        }

    );

}


async function searchCity() {

    const input =
        document.getElementById(
            "cityInput"
        );

    const city =
        input.value.trim();


    if (!city) {

        toast(
            "Enter a city name first."
        );

        return;

    }


    const status =
        document.getElementById(
            "gpsStatus"
        );


    status.textContent =
        "Searching city...";


    try {

        const url =
            "https://nominatim.openstreetmap.org/search"
            +
            "?format=json"
            +
            "&limit=1"
            +
            "&q="
            +
            encodeURIComponent(city);


        const response =
            await fetch(url);


        const data =
            await response.json();


        if (
            !data ||
            data.length === 0
        ) {

            status.innerHTML =
                `
                <span class="red">
                    City not found.
                </span>
                `;

            return;

        }


        const latitude =
            Number(
                data[0].lat
            );

        const longitude =
            Number(
                data[0].lon
            );


        localStorage.setItem(
            "userLatitude",
            latitude
        );

        localStorage.setItem(
            "userLongitude",
            longitude
        );


        document.getElementById(
            "manualLat"
        ).value =
            latitude.toFixed(6);


        document.getElementById(
            "manualLon"
        ).value =
            longitude.toFixed(6);


        status.innerHTML = `

            <span class="green">

                ✓ ${city} located

            </span>

        `;


        calculateLocationRisk(
            latitude,
            longitude
        );


    }

    catch (error) {

        console.error(error);

        status.innerHTML = `

            <span class="red">

                ❌ City search failed.

            </span>

        `;

    }

}


function checkManualLocation() {

    const latitude =
        Number(
            document.getElementById(
                "manualLat"
            ).value
        );

    const longitude =
        Number(
            document.getElementById(
                "manualLon"
            ).value
        );


    if (
        Number.isNaN(latitude) ||
        Number.isNaN(longitude)
    ) {

        toast(
            "Enter valid latitude and longitude."
        );

        return;

    }


    if (
        latitude < -90 ||
        latitude > 90 ||
        longitude < -180 ||
        longitude > 180
    ) {

        toast(
            "Coordinates are outside valid range."
        );

        return;

    }


    localStorage.setItem(
        "userLatitude",
        latitude
    );

    localStorage.setItem(
        "userLongitude",
        longitude
    );


    calculateLocationRisk(
        latitude,
        longitude
    );

}


async function calculateLocationRisk(
    latitude,
    longitude
) {

    const result =
        document.getElementById(
            "locationResult"
        );


    result.innerHTML = `

        <div class="small">

            Calculating prototype risk...

        </div>

    `;


    try {

        const response =
            await fetch(
                `/api/location/${latitude}/${longitude}`
            );


        const data =
            await response.json();


        if (data.error) {

            throw new Error(
                data.error
            );

        }


        displayLocationRisk(
            data
        );

    }

    catch (error) {

        console.error(error);

        result.innerHTML = `

            <div class="red">

                ❌ Unable to calculate
                location risk.

            </div>

        `;

    }

}


function displayLocationRisk(data) {

    const result =
        document.getElementById(
            "locationResult"
        );


    const level =
        String(
            data.risk_level || "Minimal"
        ).toLowerCase();


    let color =
        "green";


    if (
        level === "extreme"
    ) {

        color = "red";

    }

    else if (
        level === "high"
    ) {

        color = "orange";

    }

    else if (
        level === "moderate"
    ) {

        color = "yellow";

    }


    result.innerHTML = `

        <div
            class="
                location-risk
                ${level}
            "
        >

            <div class="label">

                PROTOTYPE RISK SCORE

            </div>


            <div
                class="
                    value
                    ${color}
                "
            >

                ${data.risk_score}

                / 100

            </div>


            <div
                class="
                    title
                    ${color}
                "
            >

                ${data.risk_level}

            </div>


            <p class="small">

                ${data.message}

            </p>


            <div class="rows">


                <div class="row">

                    <span>
                        Your Location
                    </span>

                    <b>

                        ${Number(
                            data.user_latitude
                        ).toFixed(4)}
                        ,
                        ${Number(
                            data.user_longitude
                        ).toFixed(4)}

                    </b>

                </div>


                <div class="row">

                    <span>
                        Current Event Distance
                    </span>

                    <b>

                        ${data.current_distance_km}
                        km

                    </b>

                </div>


                <div class="row">

                    <span>
                        Predicted Event Distance
                    </span>

                    <b class="purple">

                        ${data.predicted_distance_km}
                        km

                    </b>

                </div>


                <div class="row">

                    <span>
                        Event Wind
                    </span>

                    <b class="red">

                        ${data.current_event.maximum_wind_ms}
                        m/s

                    </b>

                </div>


                <div class="row">

                    <span>
                        Movement
                    </span>

                    <b>

                        ${data.movement_km}
                        km

                    </b>

                </div>


            </div>

        </div>

    `;

}


function loadSavedLocation() {

    const latitude =
        localStorage.getItem(
            "userLatitude"
        );

    const longitude =
        localStorage.getItem(
            "userLongitude"
        );


    if (
        latitude &&
        longitude
    ) {

        document.getElementById(
            "manualLat"
        ).value =
            Number(latitude).toFixed(6);


        document.getElementById(
            "manualLon"
        ).value =
            Number(longitude).toFixed(6);


        calculateLocationRisk(
            Number(latitude),
            Number(longitude)
        );

    }

}


function clearLocation() {

    localStorage.removeItem(
        "userLatitude"
    );

    localStorage.removeItem(
        "userLongitude"
    );


    const result =
        document.getElementById(
            "locationResult"
        );


    if (result) {

        result.innerHTML = `

            <div class="small">

                Location cleared.

            </div>

        `;

    }


    const lat =
        document.getElementById(
            "manualLat"
        );

    const lon =
        document.getElementById(
            "manualLon"
        );


    if (lat) lat.value = "";

    if (lon) lon.value = "";

}


/* ============================================================
   DOWNSCALING
   ============================================================ */

function renderDownscaling() {

    const standard =
        DOWNSCALING.models
            ?.standard_cnn || {};

    const extreme =
        DOWNSCALING.models
            ?.extreme_cnn || {};

    const physics =
        DOWNSCALING.models
            ?.physics_extreme_cnn || {};


    document.getElementById(
        "content"
    ).innerHTML = `

        <div class="card">

            <div class="head">

                <div>

                    <div class="title">
                        EXTREME-AWARE DOWNSCALING
                    </div>

                    <div class="small">
                        Prototype spatial enhancement
                    </div>

                </div>

                <span class="badge">
                    CNN PROTOTYPE
                </span>

            </div>


            <div class="down-grid section">


                <div class="card">

                    <div class="title">
                        LOW-RES INPUT
                    </div>

                    <div class="heat section"></div>

                    <div class="small section">
                        10 × 10 prototype field
                    </div>

                </div>


                <div class="card">

                    <div class="title">
                        STANDARD CNN
                    </div>

                    <div
                        class="heat section"
                        style="
                            filter:
                            blur(3px)
                            saturate(.8);
                        "
                    ></div>

                    <div class="small section">
                        Baseline model
                    </div>

                </div>


                <div class="card">

                    <div class="title">
                        EXTREME-AWARE MODEL
                    </div>

                    <div
                        class="heat section"
                        style="
                            filter:
                            saturate(1.6)
                            contrast(1.15);
                        "
                    ></div>

                    <div class="small section">
                        Extreme-preserving prototype
                    </div>

                </div>

            </div>


            <div class="grid3 section">


                <div class="card">

                    <div class="title">
                        STANDARD CNN
                    </div>

                    <div class="row">

                        <span>
                            MSE
                        </span>

                        <b>
                            ${Number(
                                standard.mse || .548185
                            ).toFixed(6)}
                        </b>

                    </div>

                    <div class="row">

                        <span>
                            MAE
                        </span>

                        <b>
                            ${Number(
                                standard.mae || .589521
                            ).toFixed(6)}
                        </b>

                    </div>

                    <div class="row">

                        <span>
                            Mean max error
                        </span>

                        <b>
                            ${Number(
                                standard.mean_max_error || .621413
                            ).toFixed(6)}
                        </b>

                    </div>

                </div>


                <div class="card">

                    <div class="title">
                        EXTREME-AWARE CNN
                    </div>

                    <div class="row">

                        <span>
                            MSE
                        </span>

                        <b>
                            ${Number(
                                extreme.mse || .698831
                            ).toFixed(6)}
                        </b>

                    </div>

                    <div class="row">

                        <span>
                            MAE
                        </span>

                        <b>
                            ${Number(
                                extreme.mae || .661480
                            ).toFixed(6)}
                        </b>

                    </div>

                    <div class="row">

                        <span>
                            Mean max error
                        </span>

                        <b class="green">
                            ${Number(
                                extreme.mean_max_error || .521488
                            ).toFixed(6)}
                        </b>

                    </div>

                </div>


                <div class="card">

                    <div class="title">
                        MAX PRESERVATION
                    </div>

                    <div class="value green">
                        59%
                    </div>

                    <div class="small">
                        Better in 118 / 200 samples
                    </div>

                    <div class="row">

                        <span>
                            Extreme pixel MAE
                        </span>

                        <b class="green">
                            ${Number(
                                extreme.extreme_pixel_mae || .587120
                            ).toFixed(6)}
                        </b>

                    </div>

                </div>

            </div>


            <p class="small section">

                These results come from the current
                synthetic prototype dataset and are
                not operational NWP validation results.

                The intended architecture uses
                diffusion-based downscaling.

            </p>

        </div>

    `;

}


/* ============================================================
   RISK PAGE
   ============================================================ */

function renderRisk() {

    document.getElementById(
        "content"
    ).innerHTML = `

        <div class="grid2">


            <div class="card">

                <div class="title">
                    RISK ASSESSMENT
                </div>


                <div class="risk-meter">

                    <div class="risk-ring"></div>

                    <div class="risk-number">

                        <b>
                            ${ALERT.intensity_score}
                        </b>

                        <br>

                        <span class="red">
                            EXTREME
                        </span>

                    </div>

                </div>


                <div
                    class="small"
                    style="text-align:center"
                >

                    Prototype risk score

                </div>

            </div>


            <div class="card">

                <div class="title">
                    RISK ENGINE
                </div>


                <div class="rows">

                    <div class="row">

                        <span>
                            Maximum Wind
                        </span>

                        <b>
                            62.91 m/s
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Movement
                        </span>

                        <b>
                            103.90 km
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Direction
                        </span>

                        <b>
                            NorthEast
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Affected Radius
                        </span>

                        <b>
                            5 km
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Intensity
                        </span>

                        <b class="red">
                            89.87 / 100
                        </b>

                    </div>

                </div>


                <p class="small section">

                    The prototype risk engine combines
                    anomaly intensity, movement and
                    spatial proximity for demonstration
                    risk scoring.

                </p>

            </div>

        </div>


        <div class="card section">

            <div class="title">
                EMERGENCY ALERT CENTER
            </div>


            <div class="grid3 section">


                <div>

                    <div class="label">
                        STATUS
                    </div>

                    <div class="value red">
                        ● ACTIVE
                    </div>

                </div>


                <div>

                    <div class="label">
                        ALERT ID
                    </div>

                    <div class="value">
                        EWA-001
                    </div>

                </div>


                <div>

                    <div class="label">
                        SEVERITY
                    </div>

                    <div class="value red">
                        EXTREME
                    </div>

                </div>

            </div>


            <div class="grid3 section">


                <div class="card">

                    <div class="label">
                        LOCATION
                    </div>

                    <div class="value">
                        18.09° N
                    </div>

                    <div class="small">
                        89.36° E
                    </div>

                </div>


                <div class="card">

                    <div class="label">
                        WIND
                    </div>

                    <div class="value red">
                        62.91 m/s
                    </div>

                </div>


                <div class="card">

                    <div class="label">
                        RADIUS
                    </div>

                    <div class="value orange">
                        5 km
                    </div>

                </div>

            </div>


            <p class="insight section">

                Coordinate-based prototype alert
                generated for an extreme anomaly.

                Real operational alert thresholds
                require meteorological validation.

            </p>


            <div class="actions section">

                <button
                    class="btn"
                    onclick="showPage('tracking')"
                >
                    VIEW TRAJECTORY
                </button>

                <button
                    class="btn"
                    onclick="exportAlert()"
                >
                    EXPORT ALERT
                </button>

                <button
                    class="btn"
                    onclick="copyAlertJSON()"
                >
                    COPY JSON
                </button>

            </div>

        </div>

    `;

}


/* ============================================================
   PIPELINE PAGE
   ============================================================ */

const PIPELINE_STAGES = [

    "Weather Data Ingestion",

    "Extreme Anomaly Detection",

    "Anomaly Region Extraction",

    "Dynamic Bounding Boxes",

    "Weather Event Records",

    "Trajectory Dataset",

    "Spatio-Temporal Graph",

    "AI Trajectory Prediction",

    "Risk Classification",

    "Emergency Alert Generation"

];


function renderPipeline() {

    document.getElementById(
        "content"
    ).innerHTML = `

        <div class="pipeline-wrapper">


            <div class="card">

                <div class="head">

                    <div>

                        <div class="title">
                            END-TO-END AI PIPELINE
                        </div>

                        <div class="small">
                            Visual execution of the
                            prototype processing flow
                        </div>

                    </div>

                    <span
                        id="pipelinePercent"
                        class="badge"
                    >
                        0%
                    </span>

                </div>


                <div
                    class="progress-wrap"
                >

                    <div
                        id="pipelineProgress"
                        class="progress-bar"
                    ></div>

                </div>


                <div
                    id="pipelineStages"
                    class="section"
                >

                </div>

            </div>


            <div class="card">

                <div class="title">
                    PIPELINE STATUS
                </div>


                <div
                    id="pipelineStatusText"
                    class="value blue"
                >

                    READY

                </div>


                <div class="small">

                    The dashboard animation
                    demonstrates the flow without
                    rerunning the Python pipeline.

                </div>


                <div class="section">

                    <div class="row">

                        <span>
                            Data
                        </span>

                        <b class="green">
                            READY
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Detection
                        </span>

                        <b class="green">
                            READY
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Tracking
                        </span>

                        <b class="green">
                            READY
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            AI Prediction
                        </span>

                        <b class="green">
                            READY
                        </b>

                    </div>


                    <div class="row">

                        <span>
                            Alerts
                        </span>

                        <b class="green">
                            READY
                        </b>

                    </div>

                </div>


                <button
                    class="btn primary section"
                    onclick="runPipelineAnimation()"
                >

                    ▶ RUN VISUAL PIPELINE

                </button>

            </div>

        </div>

    `;


    createPipelineStages();

}


function createPipelineStages() {

    const container =
        document.getElementById(
            "pipelineStages"
        );


    if (!container) {
        return;
    }


    container.innerHTML =
        PIPELINE_STAGES
            .map(
                (stage, index) => `

                    <div
                        id="stage-${index}"
                        class="pipeline-stage"
                    >

                        <div class="pipeline-number">

                            ${String(
                                index + 1
                            ).padStart(2, "0")}

                        </div>


                        <div>

                            <b>
                                ${stage}
                            </b>

                        </div>


                        <div
                            id="status-${index}"
                            class="pipeline-status"
                        >

                            WAITING

                        </div>

                    </div>

                `
            )
            .join("");

}


function sleep(ms) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                ms
            )
    );

}


async function runPipelineAnimation() {

    createPipelineStages();


    const progress =
        document.getElementById(
            "pipelineProgress"
        );

    const percent =
        document.getElementById(
            "pipelinePercent"
        );

    const status =
        document.getElementById(
            "pipelineStatusText"
        );


    status.textContent =
        "RUNNING";


    for (
        let index = 0;
        index < PIPELINE_STAGES.length;
        index++
    ) {

        const stage =
            document.getElementById(
                `stage-${index}`
            );

        const stageStatus =
            document.getElementById(
                `status-${index}`
            );


        stage.classList.add(
            "running"
        );

        stageStatus.textContent =
            "RUNNING";


        const currentPercent =
            Math.round(
                (
                    index /
                    PIPELINE_STAGES.length
                ) * 100
            );


        progress.style.width =
            `${currentPercent}%`;


        percent.textContent =
            `${currentPercent}%`;


        await sleep(450);


        stage.classList.remove(
            "running"
        );

        stage.classList.add(
            "complete"
        );

        stageStatus.textContent =
            "COMPLETE";

    }


    progress.style.width =
        "100%";

    percent.textContent =
        "100%";

    status.textContent =
        "COMPLETED";


    toast(
        "AI pipeline demonstration completed."
    );

}


/* ============================================================
   API PAGE
   ============================================================ */

function renderAPI() {

    const example = {

        alert_id:
            "EWA-001",

        status:
            "ACTIVE",

        severity:
            "Extreme",

        location: {

            latitude:
                18.0909,

            longitude:
                89.3636,

            affected_radius_km:
                5

        },

        maximum_wind_speed:
            62.91,

        intensity_score:
            89.87,

        movement_km:
            103.90,

        direction:
            "NorthEast"

    };


    document.getElementById(
        "content"
    ).innerHTML = `

        <div class="grid2">


            <div class="card">

                <div class="title">
                    REST ENDPOINTS
                </div>


                <div class="row">

                    <b>
                        GET /api/health
                    </b>

                    <span class="green">
                        200 OK
                    </span>

                </div>


                <div class="row">

                    <b>
                        GET /api/dashboard
                    </b>

                    <span class="green">
                        200 OK
                    </span>

                </div>


                <div class="row">

                    <b>
                        GET /api/alert
                    </b>

                    <span class="green">
                        200 OK
                    </span>

                </div>


                <div class="row">

                    <b>
                        GET /api/trajectory
                    </b>

                    <span class="green">
                        200 OK
                    </span>

                </div>


                <div class="row">

                    <b>
                        GET /api/gnn
                    </b>

                    <span class="green">
                        200 OK
                    </span>

                </div>


                <div class="row">

                    <b>
                        GET /api/downscaling
                    </b>

                    <span class="green">
                        200 OK
                    </span>

                </div>


                <div class="row">

                    <b>
                        GET /api/pipeline-status
                    </b>

                    <span class="green">
                        200 OK
                    </span>

                </div>


                <button
                    class="btn section"
                    onclick="copyAlertJSON()"
                >

                    COPY ALERT JSON

                </button>

            </div>


            <div class="card">

                <div class="title">
                    EXAMPLE ALERT JSON
                </div>


                <pre
                    id="apiJSON"
                    class="code section"
                ></pre>

            </div>

        </div>

    `;


    document.getElementById(
        "apiJSON"
    ).textContent =
        JSON.stringify(
            example,
            null,
            2
        );

}


/* ============================================================
   ABOUT PAGE
   ============================================================ */

function renderAbout() {

    document.getElementById(
        "content"
    ).innerHTML = `

        <div class="grid2">


            <div class="card">

                <div class="title">
                    PROJECT
                </div>


                <div class="row">

                    <span>
                        Problem Statement
                    </span>

                    <b>
                        26078
                    </b>

                </div>


                <div class="row">

                    <span>
                        Organization
                    </span>

                    <b>
                        MoES / NCMRWF
                    </b>

                </div>


                <div class="row">

                    <span>
                        Category
                    </span>

                    <b>
                        Software
                    </b>

                </div>


                <div class="row">

                    <span>
                        Theme
                    </span>

                    <b>
                        Smart Automation
                    </b>

                </div>


                <div class="row">

                    <span>
                        Team
                    </span>

                    <b>
                        Futureforge DS
                    </b>

                </div>


                <p class="insight section">

                    ExtremeWeather AI is a
                    proof-of-concept platform for
                    detecting, tracking and assessing
                    extreme weather anomalies in
                    medium-range forecast fields.

                </p>

            </div>


            <div class="card">

                <div class="title">
                    TEAM
                </div>


                <div class="chips section">

                    <span class="chip">
                        KRITHIKA S PRAKASH
                    </span>

                    <span class="chip">
                        S JITHENDRA PATEL
                    </span>

                    <span class="chip">
                        ANUSHA C
                    </span>

                    <span class="chip">
                        CHAITANYA K V
                    </span>

                    <span class="chip">
                        AMRUTHA S
                    </span>

                    <span class="chip">
                        JYOTHIKA PATTIPATI
                    </span>

                </div>


                <div class="title section">
                    INTENDED DATA SOURCES
                </div>


                <div class="chips section">

                    <span class="chip">
                        ERA5
                    </span>

                    <span class="chip">
                        IMDAA
                    </span>

                    <span class="chip">
                        NCUM
                    </span>

                    <span class="chip">
                        NEPS-G
                    </span>

                </div>


                <div class="title section">
                    TECHNOLOGY STACK
                </div>


                <div class="chips section">

                    <span class="chip">
                        Python
                    </span>

                    <span class="chip">
                        NumPy
                    </span>

                    <span class="chip">
                        NetCDF
                    </span>

                    <span class="chip">
                        Xarray
                    </span>

                    <span class="chip">
                        PyTorch
                    </span>

                    <span class="chip">
                        GNN
                    </span>

                    <span class="chip">
                        CNN
                    </span>

                    <span class="chip">
                        Flask
                    </span>

                    <span class="chip">
                        REST API
                    </span>

                    <span class="chip">
                        HTML
                    </span>

                    <span class="chip">
                        CSS
                    </span>

                    <span class="chip">
                        JavaScript
                    </span>

                </div>


                <p class="small section">

                    Real operational integration is a
                    future validation stage of the
                    prototype.

                </p>

            </div>

        </div>

    `;

}


/* ============================================================
   UTILITIES
   ============================================================ */

function formatCoord(
    latitude,
    longitude
) {

    const lat =
        Number(latitude);

    const lon =
        Number(longitude);


    if (
        Number.isNaN(lat) ||
        Number.isNaN(lon)
    ) {

        return "—";

    }


    return `
        ${Math.abs(lat).toFixed(2)}°
        ${lat >= 0 ? "N" : "S"},
        ${Math.abs(lon).toFixed(2)}°
        ${lon >= 0 ? "E" : "W"}
    `;

}


function copyAlertJSON() {

    navigator.clipboard
        .writeText(
            JSON.stringify(
                ALERT,
                null,
                2
            )
        )
        .then(
            () =>
                toast(
                    "Alert JSON copied."
                )
        )
        .catch(
            () =>
                toast(
                    "Copy unavailable."
                )
        );

}


function exportAlert() {

    const blob =
        new Blob(
            [
                JSON.stringify(
                    ALERT,
                    null,
                    2
                )
            ],
            {
                type:
                    "application/json"
            }
        );


    const url =
        URL.createObjectURL(
            blob
        );


    const link =
        document.createElement(
            "a"
        );


    link.href = url;

    link.download =
        "EWA-001-alert.json";

    link.click();


    URL.revokeObjectURL(
        url
    );


    toast(
        "Alert JSON exported."
    );

}


function togglePresentation() {

    document.body.classList.toggle(
        "presentation"
    );


    toast(
        document.body.classList.contains(
            "presentation"
        )
            ? "Presentation mode enabled."
            : "Presentation mode disabled."
    );

}


function showInfo() {

    document
        .getElementById(
            "modal"
        )
        .classList.remove(
            "hidden"
        );

}


function closeInfo() {

    document
        .getElementById(
            "modal"
        )
        .classList.add(
            "hidden"
        );

}


function toast(message) {

    const element =
        document.createElement(
            "div"
        );


    element.className =
        "toast";

    element.textContent =
        message;


    document.body.appendChild(
        element
    );


    setTimeout(
        () =>
            element.remove(),
        1800
    );

}


/* ============================================================
   INITIALIZATION
   ============================================================ */

function initializeDashboard() {

    buildNavigation();

    showPage(
        "dashboard"
    );

}


window.addEventListener(
    "resize",
    () => {

        const tracking =
            document.getElementById(
                "trackingSVG"
            );

        if (tracking) {

            renderTrackingMap();

        }

    }
);


initializeDashboard();


</script>

</body>

</html>
"""


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("=" * 60)
    print(" EXTREMEWEATHER AI DASHBOARD")
    print("=" * 60)
    print("")
    print("Server: http://127.0.0.1:5050")
    print("Mode: Prototype")
    print("")
    print("No PyTorch import in dashboard.")
    print("No external map library.")
    print("AI Tracking visualization enabled.")
    print("=" * 60)
    print("")

    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False
    )