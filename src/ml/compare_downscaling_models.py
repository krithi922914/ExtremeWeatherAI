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
# Select Same Sample
# ==========================================

sample_index = 150

input_sample = X[
    sample_index:sample_index + 1
]

target_sample = Y[
    sample_index:sample_index + 1
]


# ==========================================
# Generate Predictions
# ==========================================

with torch.no_grad():

    original_prediction = original_model(
        input_sample
    )

    extreme_prediction = extreme_model(
        input_sample
    )


# ==========================================
# Calculate Metrics
# ==========================================

original_mse = torch.mean(
    (original_prediction - target_sample) ** 2
).item()

extreme_mse = torch.mean(
    (extreme_prediction - target_sample) ** 2
).item()


original_mae = torch.mean(
    torch.abs(original_prediction - target_sample)
).item()

extreme_mae = torch.mean(
    torch.abs(extreme_prediction - target_sample)
).item()


# ==========================================
# Maximum Values
# ==========================================

target_max = torch.max(target_sample).item()

original_max = torch.max(
    original_prediction
).item()

extreme_max = torch.max(
    extreme_prediction
).item()


# ==========================================
# Print Comparison
# ==========================================

print("Downscaling Model Comparison")
print("=" * 50)

print("\nTarget Maximum:")
print(f"{target_max:.4f}")

print("\nOriginal CNN:")
print(f"MSE          : {original_mse:.6f}")
print(f"MAE          : {original_mae:.6f}")
print(f"Maximum Value: {original_max:.4f}")

print("\nExtreme-Preserving CNN:")
print(f"MSE          : {extreme_mse:.6f}")
print(f"MAE          : {extreme_mae:.6f}")
print(f"Maximum Value: {extreme_max:.4f}")


# ==========================================
# Extreme Value Error
# ==========================================

original_extreme_error = abs(
    target_max - original_max
)

extreme_model_error = abs(
    target_max - extreme_max
)


print("\nExtreme Value Error")
print("-" * 50)

print(
    f"Original CNN             : "
    f"{original_extreme_error:.4f}"
)

print(
    f"Extreme-Preserving CNN   : "
    f"{extreme_model_error:.4f}"
)


# ==========================================
# Visualization
# ==========================================

input_image = input_sample[0, 0].numpy()

target_image = target_sample[0, 0].numpy()

original_image = (
    original_prediction[0, 0]
    .numpy()
)

extreme_image = (
    extreme_prediction[0, 0]
    .numpy()
)


plt.figure(figsize=(20, 5))


# Input
plt.subplot(1, 4, 1)

plt.imshow(
    input_image,
    origin="lower"
)

plt.title("Coarse Input\n10 × 10")

plt.colorbar()


# Ground Truth
plt.subplot(1, 4, 2)

plt.imshow(
    target_image,
    origin="lower"
)

plt.title("Ground Truth\n20 × 20")

plt.colorbar()


# Original CNN
plt.subplot(1, 4, 3)

plt.imshow(
    original_image,
    origin="lower"
)

plt.title("Original CNN\nMSE Loss")

plt.colorbar()


# Extreme CNN
plt.subplot(1, 4, 4)

plt.imshow(
    extreme_image,
    origin="lower"
)

plt.title(
    "Extreme-Preserving CNN\nExtreme-Aware Loss"
)

plt.colorbar()


plt.tight_layout()

plt.show()