import csv
import numpy as np

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
            "longitude": float(row["longitude"]),
            "wind": float(row["max_wind"]),
            "intensity": float(row["intensity_score"])
        })


# ---------------------------------------
# STEP 2: Create temporal edges
# ---------------------------------------

temporal_edges = []

for i in range(len(nodes) - 1):

    temporal_edges.append((i, i + 1))


# ---------------------------------------
# STEP 3: Create spatial edges
# ---------------------------------------

spatial_edges = []

# Nodes within this distance will be considered
# geographically connected.
distance_threshold = 250.0


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


for i in range(len(nodes)):

    for j in range(i + 1, len(nodes)):

        distance = calculate_distance(
            nodes[i]["latitude"],
            nodes[i]["longitude"],
            nodes[j]["latitude"],
            nodes[j]["longitude"]
        )

        if distance <= distance_threshold:

            spatial_edges.append(
                (i, j, distance)
            )


# ---------------------------------------
# STEP 4: Display graph
# ---------------------------------------

print("Spatio-Temporal Weather Graph")
print("=" * 50)

print()
print("Total Nodes:", len(nodes))

print(
    "Temporal Edges:",
    len(temporal_edges)
)

print(
    "Spatial Edges:",
    len(spatial_edges)
)


# ---------------------------------------
# Temporal connections
# ---------------------------------------

print()
print("TEMPORAL EDGES")
print("-" * 50)

for edge in temporal_edges:

    print(
        f"Node {edge[0]} ---> Node {edge[1]}"
    )


# ---------------------------------------
# Spatial connections
# ---------------------------------------

print()
print("SPATIAL EDGES")
print("-" * 50)

for edge in spatial_edges:

    print(
        f"Node {edge[0]} <--> Node {edge[1]} "
        f"| Distance: {edge[2]:.2f} km"
    )


# ---------------------------------------
# Combined graph
# ---------------------------------------

print()
print("GRAPH SUMMARY")
print("-" * 50)

print(
    "Temporal relationships:",
    len(temporal_edges)
)

print(
    "Spatial relationships:",
    len(spatial_edges)
)

print()
print(
    "Spatio-temporal graph created successfully!"
)