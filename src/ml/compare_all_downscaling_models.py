import torch
import matplotlib.pyplot as plt

from ai_downscaling_model import WeatherDownscalingCNN


# ==========================================
# Load Dataset
# ==========================================

data = torch.load(
    "data/processed/downscaling_dataset.pt",
    map_location="cpu"
)

X = data["inputs"]
Y = data["targets"]


# ==========================================
# Function to Load Model
# ==========================================

def load_model(path):

    model = WeatherDownscalingCNN()

    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu"
        )
    )

    model.eval()

    return model


# ==========================================
# Load All Three Models
# ==========================================

original_model = load_model(
    "models/weather_downscaling_cnn.pth"
)

extreme_model = load_model(
    "models/extreme_weather_downscaling_cnn.pth"
)

physics_model = load_model(
    "models/physics_extreme_downscaling_cnn.pth"
)


# ==========================================
# Generate Predictions
# ==========================================

with torch.no_grad():

    original_prediction = original_model(X)

    extreme_prediction = extreme_model(X)

    physics_prediction = physics_model(X)


# ==========================================
# Overall Metrics
# ==========================================

def calculate_metrics(prediction, target):

    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    mae = torch.mean(
        torch.abs(prediction - target)
    ).item()

    return mse, mae


original_mse, original_mae = calculate_metrics(
    original_prediction,
    Y
)

extreme_mse, extreme_mae = calculate_metrics(
    extreme_prediction,
    Y
)

physics_mse, physics_mae = calculate_metrics(
    physics_prediction,
    Y
)


# ==========================================
# Maximum Intensity Analysis
# ==========================================

target_max = torch.amax(
    Y,
    dim=(1, 2, 3)
)

original_max = torch.amax(
    original_prediction,
    dim=(1, 2, 3)
)

extreme_max = torch.amax(
    extreme_prediction,
    dim=(1, 2, 3)
)

physics_max = torch.amax(
    physics_prediction,
    dim=(1, 2, 3)
)


original_max_error = torch.abs(
    target_max - original_max
)

extreme_max_error = torch.abs(
    target_max - extreme_max
)

physics_max_error = torch.abs(
    target_max - physics_max
)


# ==========================================
# Extreme Pixel Error
# ==========================================

threshold = 50.0

extreme_mask = Y >= threshold


original_extreme_mae = torch.mean(
    torch.abs(
        original_prediction[extreme_mask]
        - Y[extreme_mask]
    )
).item()

extreme_model_extreme_mae = torch.mean(
    torch.abs(
        extreme_prediction[extreme_mask]
        - Y[extreme_mask]
    )
).item()

physics_extreme_mae = torch.mean(
    torch.abs(
        physics_prediction[extreme_mask]
        - Y[extreme_mask]
    )
).item()


# ==========================================
# Print Results
# ==========================================

print("Three-Model Downscaling Comparison")
print("=" * 65)

print("\nOverall Reconstruction Error")
print("-" * 65)

print(
    f"Original CNN        | "
    f"MSE: {original_mse:.6f} | "
    f"MAE: {original_mae:.6f}"
)

print(
    f"Extreme CNN         | "
    f"MSE: {extreme_mse:.6f} | "
    f"MAE: {extreme_mae:.6f}"
)

print(
    f"Physics + Extreme   | "
    f"MSE: {physics_mse:.6f} | "
    f"MAE: {physics_mae:.6f}"
)


print("\nMaximum Intensity Error")
print("-" * 65)

print(
    f"Original CNN        : "
    f"{torch.mean(original_max_error).item():.6f}"
)

print(
    f"Extreme CNN         : "
    f"{torch.mean(extreme_max_error).item():.6f}"
)

print(
    f"Physics + Extreme   : "
    f"{torch.mean(physics_max_error).item():.6f}"
)


print("\nMedian Maximum Intensity Error")
print("-" * 65)

print(
    f"Original CNN        : "
    f"{torch.median(original_max_error).item():.6f}"
)

print(
    f"Extreme CNN         : "
    f"{torch.median(extreme_max_error).item():.6f}"
)

print(
    f"Physics + Extreme   : "
    f"{torch.median(physics_max_error).item():.6f}"
)


print("\nExtreme Pixel Error")
print("-" * 65)

print(
    f"Original CNN        : "
    f"{original_extreme_mae:.6f}"
)

print(
    f"Extreme CNN         : "
    f"{extreme_model_extreme_mae:.6f}"
)

print(
    f"Physics + Extreme   : "
    f"{physics_extreme_mae:.6f}"
)


# ==========================================
# Maximum Value Comparison
# ==========================================

print("\nAverage Predicted Maximum")
print("-" * 65)

print(
    f"Target              : "
    f"{torch.mean(target_max).item():.6f}"
)

print(
    f"Original CNN        : "
    f"{torch.mean(original_max).item():.6f}"
)

print(
    f"Extreme CNN         : "
    f"{torch.mean(extreme_max).item():.6f}"
)

print(
    f"Physics + Extreme   : "
    f"{torch.mean(physics_max).item():.6f}"
)


# ==========================================
# Visualization
# ==========================================

sample_index = 150

input_image = X[
    sample_index, 0
].numpy()

target_image = Y[
    sample_index, 0
].numpy()

original_image = original_prediction[
    sample_index, 0
].numpy()

extreme_image = extreme_prediction[
    sample_index, 0
].numpy()

physics_image = physics_prediction[
    sample_index, 0
].numpy()


plt.figure(figsize=(25, 5))


plt.subplot(1, 5, 1)

plt.imshow(
    input_image,
    origin="lower"
)

plt.title("Coarse Input\n10 × 10")

plt.colorbar()


plt.subplot(1, 5, 2)

plt.imshow(
    target_image,
    origin="lower"
)

plt.title("Ground Truth\n20 × 20")

plt.colorbar()


plt.subplot(1, 5, 3)

plt.imshow(
    original_image,
    origin="lower"
)

plt.title("Original CNN")

plt.colorbar()


plt.subplot(1, 5, 4)

plt.imshow(
    extreme_image,
    origin="lower"
)

plt.title("Extreme-Preserving CNN")

plt.colorbar()


plt.subplot(1, 5, 5)

plt.imshow(
    physics_image,
    origin="lower"
)

plt.title("Physics + Extreme CNN")

plt.colorbar()


plt.tight_layout()

plt.show()