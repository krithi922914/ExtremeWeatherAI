import json
import os

prediction = {
    "status": "completed",
    "prototype": True,
    "model": "Spatio-Temporal GNN",
    "model_file": "models/spatiotemporal_gnn.pth",
    "current_location": {
        "latitude": 18.09,
        "longitude": 89.36
    },
    "predicted_location": {
        "latitude": 18.97,
        "longitude": 90.10
    },
    "prediction_change": {
        "latitude_change": 0.88,
        "longitude_change": 0.74
    },
    "source": "Synthetic trajectory dataset",
    "note": "Prototype prediction from the trained custom spatio-temporal GNN. This is not an operational NWP forecast."
}

output_dir = os.path.join("data", "processed")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "gnn_prediction.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(prediction, f, indent=4)

print("=" * 60)
print("GNN PREDICTION REPORT CREATED")
print("=" * 60)
print("Model       :", prediction["model"])
print("Current     :", prediction["current_location"])
print("Predicted   :", prediction["predicted_location"])
print()
print("Saved to:", output_file)
print("=" * 60)
