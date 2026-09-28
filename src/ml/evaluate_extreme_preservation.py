import torch
import numpy as np

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
# Load Original CNN
# ==========================================

original_model = WeatherDownscalingCNN()

original_model.load_state_dict(
    torch.load(
        "models/weather_downscaling_cnn.pth",
        map_location="cpu"
    )
)

original_model.eval()


# ==========================================
# Load Extreme-Preserving CNN
# ==========================================

extreme_model = WeatherDownscalingCNN()

extreme_model.load_state_dict(
    torch.load(
        "models/extreme_weather_downscaling_cnn.pth",
        map_location="cpu"
    )
)

extreme_model.eval()


# ==========================================
# Generate Predictions
# ==========================================

with torch.no_grad():

    original_predictions = original_model(X)

    extreme_predictions = extreme_model(X)


# ==========================================
# Dataset-Wide Metrics
# ==========================================

original_mse = torch.mean(
    (original_predictions - Y) ** 2
).item()

extreme_mse = torch.mean(
    (extreme_predictions - Y) ** 2
).item()


original_mae = torch.mean(
    torch.abs(original_predictions - Y)
).item()

extreme_mae = torch.mean(
    torch.abs(extreme_predictions - Y)
).item()


# ==========================================
# Maximum Value Analysis
# ==========================================

target_max = torch.amax(
    Y,
    dim=(1, 2, 3)
)

original_max = torch.amax(
    original_predictions,
    dim=(1, 2, 3)
)

extreme_max = torch.amax(
    extreme_predictions,
    dim=(1, 2, 3)
)


# Absolute error in maximum weather intensity

original_max_error = torch.abs(
    target_max - original_max
)

extreme_max_error = torch.abs(
    target_max - extreme_max
)


# ==========================================
# Compare Which Model Preserves Max Better
# ==========================================

original_better = (
    original_max_error < extreme_max_error
)

extreme_better = (
    extreme_max_error < original_max_error
)

same_error = (
    original_max_error == extreme_max_error
)


# ==========================================
# Print Results
# ==========================================

print("Dataset-Wide Extreme Preservation Evaluation")
print("=" * 60)

print(f"Number of samples: {len(X)}")


print("\nOverall Reconstruction Error")
print("-" * 60)

print(f"Original CNN MSE : {original_mse:.6f}")
print(f"Extreme CNN MSE  : {extreme_mse:.6f}")

print(f"\nOriginal CNN MAE : {original_mae:.6f}")
print(f"Extreme CNN MAE  : {extreme_mae:.6f}")


print("\nMaximum Intensity Preservation")
print("-" * 60)

print(
    f"Original CNN mean max error : "
    f"{torch.mean(original_max_error).item():.6f}"
)

print(
    f"Extreme CNN mean max error  : "
    f"{torch.mean(extreme_max_error).item():.6f}"
)


print(
    f"\nOriginal CNN median max error : "
    f"{torch.median(original_max_error).item():.6f}"
)

print(
    f"Extreme CNN median max error  : "
    f"{torch.median(extreme_max_error).item():.6f}"
)


print("\nWhich Model Preserved Maximum Intensity Better?")
print("-" * 60)

print(
    f"Original CNN better : "
    f"{torch.sum(original_better).item()} samples"
)

print(
    f"Extreme CNN better  : "
    f"{torch.sum(extreme_better).item()} samples"
)

print(
    f"Same error          : "
    f"{torch.sum(same_error).item()} samples"
)


# ==========================================
# Percentage Comparison
# ==========================================

original_percentage = (
    torch.mean(original_better.float()).item() * 100
)

extreme_percentage = (
    torch.mean(extreme_better.float()).item() * 100
)


print("\nPercentage of Samples")
print("-" * 60)

print(
    f"Original CNN better : "
    f"{original_percentage:.2f}%"
)

print(
    f"Extreme CNN better  : "
    f"{extreme_percentage:.2f}%"
)


# ==========================================
# Extreme Pixel Analysis
# ==========================================

threshold = 50.0

extreme_mask = Y >= threshold


if torch.any(extreme_mask):

    original_extreme_mae = torch.mean(
        torch.abs(
            original_predictions[extreme_mask]
            - Y[extreme_mask]
        )
    ).item()

    extreme_model_extreme_mae = torch.mean(
        torch.abs(
            extreme_predictions[extreme_mask]
            - Y[extreme_mask]
        )
    ).item()

    print("\nExtreme Pixel Error")
    print("-" * 60)

    print(
        f"Extreme pixels evaluated: "
        f"{torch.sum(extreme_mask).item()}"
    )

    print(
        f"Original CNN extreme MAE : "
        f"{original_extreme_mae:.6f}"
    )

    print(
        f"Extreme CNN extreme MAE  : "
        f"{extreme_model_extreme_mae:.6f}"
    )


print("\nEvaluation completed.")