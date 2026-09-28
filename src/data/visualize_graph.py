import csv
import numpy as np
import matplotlib.pyplot as plt

input_file = "data/processed/trajectory_dataset.csv"

nodes = []

# ---------------------------------------
# STEP 1: Load trajectory data
# ---------------------------------------

with open(input_file, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        nodes.append({
            "time": int(row["time"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"])
        })


# ---------------------------------------
# STEP 2: Distance function
# ---------------------------------------

def calculate_distance(lat1, lon1, lat2, lon2):

    lat_distance = (lat2 - lat1) * 111

    lon_distance = (
        (lon2 - lon1)
        * 111
        * np.cos(np.radians((lat1 + lat2) / 2))
    )

    return np.sqrt(
        lat_distance ** 2 +
        lon_distance ** 2
    )


# ---------------------------------------
# STEP 3: Build edges
# ---------------------------------------

temporal_edges = []

for i in range(len(nodes) - 1):
    temporal_edges.append((i, i + 1))


spatial_edges = []

distance_threshold = 250.0

for i in range(len(nodes)):

    for j in range(i + 1, len(nodes)):

        distance = calculate_distance(
            nodes[i]["latitude"],
            nodes[i]["longitude"],
            nodes[j]["latitude"],
            nodes[j]["longitude"]
        )

        if distance <= distance_threshold:

            spatial_edges.append((i, j))


# ---------------------------------------
# STEP 4: Create graph
# ---------------------------------------

plt.figure(figsize=(10, 7))

# Draw spatial edges
for i, j in spatial_edges:

    plt.plot(
        [
            nodes[i]["longitude"],
            nodes[j]["longitude"]
        ],
        [
            nodes[i]["latitude"],
            nodes[j]["latitude"]
        ],
        linewidth=1,
        alpha=0.35
    )


# Draw temporal edges
for i, j in temporal_edges:

    plt.plot(
        [
            nodes[i]["longitude"],
            nodes[j]["longitude"]
        ],
        [
            nodes[i]["latitude"],
            nodes[j]["latitude"]
        ],
        linewidth=2
    )


# Draw nodes
longitudes = [
    node["longitude"]
    for node in nodes
]

latitudes = [
    node["latitude"]
    for node in nodes
]

plt.scatter(
    longitudes,
    latitudes,
    s=100,
    zorder=3,
    label="Weather Nodes"
)


# Label each node
for node in nodes:

    plt.text(
        node["longitude"],
        node["latitude"],
        f" T{node['time']}",
        fontsize=10
    )


plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")

plt.title(
    "Spatio-Temporal Extreme Weather Graph"
)

plt.grid(True)
plt.legend()

plt.show()
