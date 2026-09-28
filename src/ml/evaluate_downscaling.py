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

print("Downscaling Evaluation")
print("=" * 45)

print("Dataset size:", X.shape[0])
print("Input shape:", X.shape)
print("Target shape:", Y.shape)


# ==========================================
# Load Trained Model
# ==========================================

model = WeatherDownscalingCNN()

model.load_state_dict(
    torch.load(
        "models/weather_downscaling_cnn.pth",
        map_location="cpu"
    )
)

model.eval()

print("\nModel loaded successfully!")


# ==========================================
# Select a Sample
# ==========================================

sample_index = 150

input_sample = X[sample_index:sample_index + 1]
target_sample = Y[sample_index:sample_index + 1]


# ==========================================
# Prediction
# ==========================================

with torch.no_grad():

    prediction = model(input_sample)


# ==========================================
# Calculate Errors
# ==========================================

mse = torch.mean(
    (prediction - target_sample) ** 2
).item()

mae = torch.mean(
    torch.abs(prediction - target_sample)
).item()


print("\nEvaluation Results")
print("-" * 45)

print(f"MSE : {mse:.6f}")
print(f"MAE : {mae:.6f}")


# ==========================================
# Extreme Value Comparison
# ==========================================

target_max = torch.max(target_sample).item()
prediction_max = torch.max(prediction).item()

print("\nExtreme Value Comparison")
print("-" * 45)

print(f"Target maximum     : {target_max:.4f}")
print(f"Predicted maximum  : {prediction_max:.4f}")


# ==========================================
# Visualization
# ==========================================

input_image = input_sample[0, 0].numpy()
target_image = target_sample[0, 0].numpy()
prediction_image = prediction[0, 0].numpy()


plt.figure(figsize=(15, 5))


# Coarse input
plt.subplot(1, 3, 1)

plt.imshow(
    input_image,
    origin="lower"
)

plt.title("Coarse Weather Input\n10 × 10")
plt.colorbar()


# Ground truth
plt.subplot(1, 3, 2)

plt.imshow(
    target_image,
    origin="lower"
)

plt.title("High-Resolution Target\n20 × 20")
plt.colorbar()


# Prediction
plt.subplot(1, 3, 3)

plt.imshow(
    prediction_image,
    origin="lower"
)

plt.title("AI Downscaled Prediction\n20 × 20")
plt.colorbar()


plt.tight_layout()

plt.show()