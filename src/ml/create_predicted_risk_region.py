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
# Latest two weather states
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


# ----------------------------------------
# Graph connection
# ----------------------------------------

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
# Create dynamic risk region
# ----------------------------------------

# Prototype radius around predicted event
latitude_radius = 1.0
longitude_radius = 1.0


south = predicted_latitude - latitude_radius
north = predicted_latitude + latitude_radius

west = predicted_longitude - longitude_radius
east = predicted_longitude + longitude_radius


# ----------------------------------------
# Display risk region
# ----------------------------------------

print("Dynamic Predicted Risk Region")
print("=" * 45)

print()

print(
    f"Predicted center: "
    f"({predicted_latitude:.2f} N, "
    f"{predicted_longitude:.2f} E)"
)

print()

print("Bounding Region:")
print(f"South: {south:.2f} N")
print(f"North: {north:.2f} N")
print(f"West : {west:.2f} E")
print(f"East : {east:.2f} E")

print()
print("Risk region generated successfully!")


# ----------------------------------------
# Visualize
# ----------------------------------------

historical_lats = [
    record["latitude"]
    for record in records
]

historical_lons = [
    record["longitude"]
    for record in records
]


plt.figure(figsize=(10, 7))


# Historical trajectory
plt.plot(
    historical_lons,
    historical_lats,
    marker="o",
    linewidth=2,
    label="Historical Trajectory"
)


# Predicted location
plt.scatter(
    predicted_longitude,
    predicted_latitude,
    s=180,
    marker="X",
    label="Predicted Event"
)


# Bounding box
box_lons = [
    west,
    east,
    east,
    west,
    west
]

box_lats = [
    south,
    south,
    north,
    north,
    south
]


plt.plot(
    box_lons,
    box_lats,
    linewidth=2,
    linestyle="--",
    label="Dynamic Risk Region"
)


plt.xlabel("Longitude (E)")
plt.ylabel("Latitude (N)")

plt.title(
    "Dynamic Predicted Extreme Weather Risk Region"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()