import csv 
import torch 
 
from spatiotemporal_gnn import SpatioTemporalWeatherGNN 
 
 
input_file = "data/processed/trajectory_dataset.csv" 
model_file = "models/spatiotemporal_gnn.pth" 
 
 
# Load trajectory data 
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
 
 
# Get latest two weather states 
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
 
 
# Bidirectional graph connection 
edge_index = torch.tensor( 
    [ 
        [0, 1], 
        [1, 0] 
    ], 
    dtype=torch.long 
).t() 
 
 
# Load trained model 
model = SpatioTemporalWeatherGNN() 
 
model.load_state_dict( 
    torch.load( 
        model_file, 
        map_location="cpu" 
    ) 
) 
 
model.eval() 
 
 
# Generate prediction 
with torch.no_grad(): 
    prediction = model( 
        node_features, 
        edge_index 
    ) 
 
 
predicted_latitude = prediction[0, 0].item() 
predicted_longitude = prediction[0, 1].item() 
 
 
# Display result 
print("Next Extreme Weather Location Prediction") 
print("=" * 50) 
 
print() 
 
print( 
    f"Current location: " 
    f"({current['latitude']:.2f} N, " 
    f"{current['longitude']:.2f} E)" 
) 
 
print( 
    f"Predicted next location: " 
    f"({predicted_latitude:.2f} N, " 
    f"{predicted_longitude:.2f} E)" 
) 
 
print() 
print("Prediction generated using trained Spatio-Temporal GNN.") 
