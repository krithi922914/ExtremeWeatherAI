# 🌦️ ExtremeWeatherAI

### AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts

**Smart India Hackathon 2026 — Problem Statement ID: 26078**

> An AI-driven prototype for detecting, tracking, predicting, downscaling, and visualizing extreme weather anomalies with risk assessment and emergency alerts.

---

## 📌 Problem Statement

Extreme weather events such as severe storms, high-wind systems, and other atmospheric anomalies can evolve rapidly across space and time.

Traditional forecasting workflows can face challenges in:

* Detecting localized extreme anomalies
* Tracking their movement across consecutive forecast periods
* Preserving extreme values during spatial downscaling
* Communicating localized risk information
* Providing an accessible visualization and alert interface

The SIH 2026 problem statement proposes an AI-based system combining **spatio-temporal graph learning** with **high-resolution downscaling** to improve the tracking and representation of extreme weather anomalies in medium-range forecasts.

---

## 💡 Proposed Solution

**ExtremeWeatherAI** is a prototype pipeline that combines:

1. Extreme weather detection
2. Anomaly-region identification
3. Dynamic bounding-box generation
4. Weather-event extraction
5. Spatio-temporal graph construction
6. AI-based trajectory prediction
7. Extreme-aware weather downscaling
8. Risk classification
9. Emergency alert generation
10. Interactive dashboard visualization

The system is designed as a foundation for integrating operational numerical weather prediction datasets and more advanced AI architectures in future development.

---

## 🎯 SIH Problem Statement

| Field                    | Details                                                       |
| ------------------------ | ------------------------------------------------------------- |
| **Problem Statement ID** | 26078                                                         |
| **Organization**         | Ministry of Earth Sciences (MoES)                             |
| **Department**           | National Centre for Medium Range Weather Forecasting (NCMRWF) |
| **Category**             | Software                                                      |
| **Theme**                | Smart Automation                                              |
| **Team**                 | Futureforge DS                                                |

---

# 🧠 System Architecture

```text
                WEATHER DATA
                     │
                     ▼
          ┌─────────────────────┐
          │ Extreme Detection   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Anomaly Region      │
          │ Detection            │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Dynamic Bounding    │
          │ Boxes                │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Event Records &     │
          │ Trajectory Dataset  │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Spatio-Temporal     │
          │ Graph               │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ GNN Trajectory      │
          │ Prediction          │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Risk Classification │
          │ & Alert Generation  │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Interactive Flask   │
          │ Dashboard           │
          └─────────────────────┘


        WEATHER FIELD
             │
             ▼
     ┌──────────────────┐
     │ AI Downscaling   │
     │ Prototype        │
     └────────┬─────────┘
              │
              ▼
      Higher-resolution
       weather field
```

---

# 🔄 AI Processing Pipeline

The prototype implements a 10-stage processing pipeline:

### 1. Weather Data Ingestion

Weather-grid data is loaded from NetCDF datasets using Python and `netCDF4`.

### 2. Extreme Weather Detection

The system identifies locations where the selected weather variable exceeds the configured extreme threshold.

### 3. Anomaly Region Detection

Neighboring extreme points are grouped into spatial anomaly regions.

### 4. Dynamic Bounding Boxes

Bounding boxes are generated around detected anomaly regions and updated over time.

### 5. Weather Event Records

Each detected event is converted into structured records containing:

* Timestamp
* Latitude
* Longitude
* Maximum wind
* Intensity score
* Intensity category
* Movement distance
* Movement direction

### 6. Trajectory Dataset

Sequential event records are converted into a trajectory dataset suitable for machine-learning experiments.

### 7. Spatio-Temporal Graph

The prototype represents weather events as graph nodes with:

* Temporal relationships
* Spatial relationships

This allows the system to represent both **where an event is** and **how it evolves over time**.

### 8. AI Trajectory Prediction

A custom graph-based neural network is trained to predict the next location of the weather anomaly.

### 9. Risk Classification

Current event characteristics and movement information are combined into a prototype risk score.

### 10. Emergency Alert Generation

The system generates structured alerts containing:

* Alert ID
* Severity
* Location
* Maximum wind
* Intensity score
* Movement
* Direction
* Affected radius
* Alert status

---

# 🤖 AI Components

## Spatio-Temporal GNN

The prototype contains a custom graph neural network for trajectory prediction.

The model uses:

* Node features
* Temporal connections
* Spatial connections
* Neighbor information
* Graph-based feature aggregation

The current implementation is a **prototype custom graph architecture** and is not yet an operational spherical/icosahedral GNN.

---

## 🌍 Extreme-Aware Downscaling

The prototype experiments with CNN-based spatial downscaling.

The current experimental setup uses:

```text
Input:  10 × 10
Output: 20 × 20
```

Three model configurations were evaluated:

* Standard CNN
* Extreme-Aware CNN
* Physics + Extreme CNN

The Extreme-Aware model introduces additional emphasis on high-intensity pixels during training.

### Prototype evaluation

| Model                 |    MSE |    MAE | Mean Max Error | Extreme Pixel MAE |
| --------------------- | -----: | -----: | -------------: | ----------------: |
| Standard CNN          | 0.5482 | 0.5895 |         0.6214 |            0.6381 |
| Extreme-Aware CNN     | 0.6988 | 0.6615 |         0.5215 |            0.5871 |
| Physics + Extreme CNN | 0.7170 | 0.6721 |         0.5633 |            0.6228 |

The Extreme-Aware CNN showed lower maximum-value error and lower extreme-pixel error than the Standard CNN in this synthetic prototype evaluation.

**Important:** these results are from the current synthetic prototype dataset and should not be interpreted as operational meteorological validation.

---

# 🚨 Risk & Alert System

The prototype converts detected weather characteristics into a risk score using:

* Current event proximity
* Predicted event proximity
* Weather intensity
* Movement characteristics

The dashboard categorizes risk into:

```text
Minimal
Low
Moderate
High
Extreme
```

The prototype also generates a localized emergency alert with a configurable affected radius.

Example alert structure:

```json
{
    "alert_id": "EWA-001",
    "status": "ACTIVE",
    "severity": "Extreme",
    "latitude": 18.0909,
    "longitude": 89.3636,
    "affected_radius_km": 5,
    "maximum_wind": 62.91,
    "intensity_score": 89.87
}
```

These thresholds and scores are currently **prototype values** and require calibration against validated meteorological observations before operational use.

---

# 🖥️ Interactive Dashboard

The project includes a Flask-based web dashboard designed to demonstrate the complete AI workflow.

## 📸 Prototype Screenshots

### Main Dashboard

![ExtremeWeatherAI Dashboard](<Screenshot 2026-09-28 152044.png>)

### AI Tracking

![AI Tracking](<Screenshot 2026-09-28 152140.png>)

### Risk & Alerts

![Risk and Alerts](<Screenshot 2026-09-28 152205.png>)

### AI Pipeline

![AI Pipeline](<Screenshot 2026-09-28 152233.png>)

## Dashboard modules

### 📊 Dashboard

Provides an overview of:

* Current weather event
* Intensity
* Movement
* Risk
* AI prediction
* Alert status

### 🌦️ Weather Monitor

Displays weather-event information and anomaly tracking data.

### 🧠 AI Tracking

Visualizes:

* Historical anomaly trajectory
* Current anomaly position
* Predicted next location
* Movement direction
* Intensity information

### 📍 My Location

Allows users to:

* Detect browser location
* Enter latitude and longitude manually
* Search for a location
* Calculate prototype location-specific risk

### 🔬 Downscaling

Displays prototype model comparison and extreme-value preservation results.

### 🚨 Risk & Alerts

Displays:

* Risk level
* Risk score
* Event location
* Wind intensity
* Movement
* Emergency alert information

### ⚙️ Pipeline

Provides a visual representation of the 10-stage AI processing pipeline.

### 🔌 API

Provides REST endpoints for accessing prototype weather, trajectory, risk, alert, prediction, and health information.

---

# 🔌 REST API

The Flask application exposes the following endpoints:

| Endpoint                               | Purpose                          |
| -------------------------------------- | -------------------------------- |
| `/`                                    | Main dashboard                   |
| `/api/dashboard`                       | Dashboard data                   |
| `/api/trajectory`                      | Weather trajectory               |
| `/api/risk`                            | Current risk information         |
| `/api/alert`                           | Emergency alert                  |
| `/api/gnn`                             | GNN prediction                   |
| `/api/downscaling`                     | Downscaling evaluation           |
| `/api/health`                          | Application health               |
| `/api/location/<latitude>/<longitude>` | Location-specific prototype risk |

---

# 🛠️ Technology Stack

## Programming

* Python
* HTML
* CSS
* JavaScript

## Machine Learning

* PyTorch
* NumPy
* SciPy

## Weather / Scientific Computing

* netCDF4
* Xarray
* Matplotlib

## Graph Learning

* Custom Graph Neural Network architecture
* Spatio-temporal graph representation

## Web Application

* Flask
* HTML/CSS/JavaScript
* Browser Geolocation API

## Data Visualization

* Matplotlib
* SVG-based dashboard visualizations

## Development

* Visual Studio Code
* Git
* GitHub

---

# 📁 Project Structure

```text
ExtremeWeatherAI/
│
├── app.py
├── .gitignore
│
├── data/
│   ├── processed/
│   │   ├── downscaling_results.json
│   │   ├── emergency_alert.json
│   │   ├── gnn_prediction.json
│   │   ├── pipeline_status.json
│   │   ├── trajectory_dataset.csv
│   │   ├── weather_alert.json
│   │   └── weather_risk.json
│   │
│   └── samples/
│       └── sample_weather.nc
│
├── models/
│   ├── extreme_weather_downscaling_cnn.pth
│   ├── physics_extreme_downscaling_cnn.pth
│   ├── spatiotemporal_gnn.pth
│   └── weather_downscaling_cnn.pth
│
├── src/
│   ├── alerts/
│   │   ├── classify_risk.py
│   │   └── create_emergency_alert.py
│   │
│   ├── data/
│   │   ├── analyze_movement.py
│   │   ├── build_spatiotemporal_graph.py
│   │   ├── build_weather_graph.py
│   │   ├── classify_intensity.py
│   │   ├── create_bounding_boxes.py
│   │   ├── create_event_records.py
│   │   ├── create_sample_weather.py
│   │   ├── create_trajectory_dataset.py
│   │   ├── detect_anomaly_region.py
│   │   ├── detect_extreme.py
│   │   ├── evaluate_baseline.py
│   │   ├── predict_trajectory.py
│   │   ├── read_sample_weather.py
│   │   ├── track_anomaly.py
│   │   ├── visualize_anomaly.py
│   │   └── visualize_graph.py
│   │
│   ├── ml/
│   │   ├── ai_downscaling_model.py
│   │   ├── gnn_model.py
│   │   ├── spatiotemporal_gnn.py
│   │   ├── train_gnn.py
│   │   ├── train_downscaling.py
│   │   ├── train_extreme_downscaling.py
│   │   ├── train_physics_downscaling.py
│   │   └── ...
│   │
│   └── run_pipeline.py
│
└── templates/
    └── dashboard.html
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/krithi922914/ExtremeWeatherAI.git
cd ExtremeWeatherAI
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

## 3. Install dependencies

Install the required Python packages:

```bash
python -m pip install numpy pandas xarray netCDF4 scipy matplotlib flask torch
```

Depending on the development environment, additional scientific or visualization libraries may be required for future extensions.

---

# ▶️ Running the Application

From the project root:

```bash
python app.py
```

The Flask development server will start locally.

Open the displayed local address in a browser, typically:

```text
http://127.0.0.1:5000
```

---

# 🔄 Running the Weather Pipeline

The main processing pipeline is:

```bash
python src/run_pipeline.py
```

The pipeline executes the major processing stages sequentially and generates updated prototype outputs in:

```text
data/processed/
```

---

# 📊 Current Prototype Capabilities

The current prototype demonstrates:

* ✅ Extreme weather anomaly detection
* ✅ Spatial anomaly-region identification
* ✅ Dynamic anomaly bounding boxes
* ✅ Weather-event extraction
* ✅ Event trajectory generation
* ✅ Spatio-temporal graph construction
* ✅ GNN-based next-location prediction
* ✅ Extreme-aware CNN downscaling experiment
* ✅ Prototype physics-informed loss experiment
* ✅ Risk scoring
* ✅ Emergency alert generation
* ✅ Browser-based location detection
* ✅ Location-specific prototype risk calculation
* ✅ Interactive Flask dashboard
* ✅ REST API endpoints
* ✅ End-to-end pipeline visualization

---

# ⚠️ Prototype Limitations

This repository represents a **research and demonstration prototype**, not an operational weather forecasting system.

Current limitations include:

* Synthetic weather data is used for the main demonstration.
* The current GNN is not yet implemented on a true spherical/icosahedral mesh.
* The current downscaling model is CNN-based rather than the proposed diffusion architecture.
* The prototype uses 10 × 10 → 20 × 20 synthetic downscaling rather than operational 12 km → 5 km data.
* Current risk thresholds are prototype values.
* The physics-informed component uses a simplified spatial-gradient constraint rather than a complete atmospheric physics model.
* Current evaluation is based on synthetic prototype data.
* Operational NWP datasets such as NCUM/NEPS-G and validated observations are not currently integrated.
* The prototype has not been validated for real-world emergency decision-making.

---

# 🚀 Future Development

The planned evolution of ExtremeWeatherAI includes:

### 🌐 Operational Weather Data

Integrate:

* NCUM deterministic forecasts
* NEPS-G ensemble forecasts
* ERA5
* IMDAA
* Validated observational datasets

### 🕸️ Spherical Graph Neural Network

Extend the current graph architecture to a spherical or icosahedral mesh representation suitable for global weather fields.

### 🎨 Diffusion
