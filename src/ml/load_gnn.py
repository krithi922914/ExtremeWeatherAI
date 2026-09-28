import torch

from spatiotemporal_gnn import SpatioTemporalWeatherGNN


model_file = "models/spatiotemporal_gnn.pth"


print("Loading trained Spatio-Temporal GNN")
print("=" * 45)


model = SpatioTemporalWeatherGNN()

model.load_state_dict(
    torch.load(
        model_file,
        map_location="cpu"
    )
)

model.eval()


print("Model loaded successfully!")
print("Model file:", model_file)
print()
print("Model architecture:")
print(model)
