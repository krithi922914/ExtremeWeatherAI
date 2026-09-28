import csv
import torch
import matplotlib.pyplot as plt

from spatiotemporal_gnn import SpatioTemporalWeatherGNN


input_file = "data/processed/trajectory_dataset.csv"
model_file = "models/spatiotemporal_gnn.pth"


# ----------------------------------------
# Load trajectory data
# ----------------------------------------

records = []

with open(input_file, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        records.append({
            "time": int(row["time"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "wind": float(row["max_wind"]),
            "intensity": float(row["intensity_score"])
        })


# ----------------------------------------
# Prepare latest two states
# ----------------------------------------

previous = records[-2]
current = records[-1]

node_features = torch.tensor(
    [
        [
            previous["latitude"],
            previous["longitude"],
            previous["wind"],
            previous["intensity"]
        ],
        [
            current["latitude"],
            current["longitude"],
            current["wind"],
            current["intensity"]
        ]
    ],
    dtype=torch.float32
)


edge_index = torch.tensor(
    [
        [0, 1],
        [1, 0]
    ],
    dtype=torch.long
).t()


# ----------------------------------------
# Load trained GNN
# ----------------------------------------

model = SpatioTemporalWeatherGNN()

model.load_state_dict(
    torch.load(
        model_file,
        map_location="cpu"
    )
)

model.eval()


# ----------------------------------------
# Predict next location
# ----------------------------------------

with torch.no_grad():

    prediction = model(
        node_features,
        edge_index
    )


predicted_latitude = prediction[0, 0].item()
predicted_longitude = prediction[0, 1].item()


# ----------------------------------------
# Historical trajectory
# ----------------------------------------

historical_lats = [
    record["latitude"]
    for record in records
]

historical_lons = [
    record["longitude"]
    for record in records
]


# ----------------------------------------
# Plot
# ----------------------------------------

plt.figure(figsize=(10, 7))

plt.plot(
    historical_lons,
    historical_lats,
    marker="o",
    linewidth=2,
    label="Historical Anomaly Trajectory"
)

plt.scatter(
    predicted_longitude,
    predicted_latitude,
    s=180,
    marker="X",
    label="GNN Predicted Location"
)

# Connect latest known point to prediction
plt.plot(
    [current["longitude"], predicted_longitude],
    [current["latitude"], predicted_latitude],
    linestyle="--",
    linewidth=2,
    label="Predicted Movement"
)


# Label historical points
for record in records:

    plt.text(
        record["longitude"],
        record["latitude"],
        f" T{record['time']}",
        fontsize=9
    )


plt.xlabel("Longitude (E)")
plt.ylabel("Latitude (N)")

plt.title(
    "Extreme Weather Trajectory + GNN Prediction"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()
