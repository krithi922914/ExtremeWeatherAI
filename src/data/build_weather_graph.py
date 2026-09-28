import csv
import numpy as np

input_file = "data/processed/trajectory_dataset.csv"

nodes = []

# ---------------------------------------
# STEP 1: Load trajectory dataset
# ---------------------------------------

with open(input_file, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        node = {
            "time": int(row["time"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "wind": float(row["max_wind"]),
            "intensity": float(row["intensity_score"])
        }

        nodes.append(node)


# ---------------------------------------
# STEP 2: Create temporal edges
# ---------------------------------------

edges = []

for i in range(len(nodes) - 1):

    edges.append((i, i + 1))


# ---------------------------------------
# STEP 3: Display graph
# ---------------------------------------

print("Spatio-Temporal Weather Graph")
print("=" * 45)

print()
print("Nodes:", len(nodes))
print("Edges:", len(edges))

print()
print("NODE INFORMATION")
print("-" * 45)

for i, node in enumerate(nodes):

    print(
        f"Node {i} | "
        f"T={node['time']} | "
        f"Location=({node['latitude']:.2f}, "
        f"{node['longitude']:.2f}) | "
        f"Wind={node['wind']:.2f} m/s | "
        f"Intensity={node['intensity']:.1f}"
    )


print()
print("TEMPORAL EDGES")
print("-" * 45)

for edge in edges:

    print(
        f"Node {edge[0]}  --->  Node {edge[1]}"
    )


# ---------------------------------------
# STEP 4: Create feature matrix
# ---------------------------------------

feature_matrix = np.array([
    [
        node["latitude"],
        node["longitude"],
        node["wind"],
        node["intensity"]
    ]
    for node in nodes
])

print()
print("FEATURE MATRIX")
print("-" * 45)

print(feature_matrix)

print()
print("Feature matrix shape:", feature_matrix.shape)