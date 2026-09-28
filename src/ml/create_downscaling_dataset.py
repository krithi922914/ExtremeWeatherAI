import os
import numpy as np
import torch


output_file = "data/processed/downscaling_dataset.pt"

os.makedirs("data/processed", exist_ok=True)


# ----------------------------------------
# Dataset settings
# ----------------------------------------

num_samples = 200

coarse_size = 10
fine_size = 20


rng = np.random.default_rng(42)


coarse_inputs = []
fine_targets = []


# ----------------------------------------
# Generate synthetic weather fields
# ----------------------------------------

for sample in range(num_samples):

    x = np.linspace(0, 1, fine_size)
    y = np.linspace(0, 1, fine_size)

    X, Y = np.meshgrid(x, y)


    # Background weather field
    weather = (
        20
        + 8 * X
        + 5 * Y
    )


    # Random extreme-weather center
    center_x = rng.uniform(0.25, 0.75)
    center_y = rng.uniform(0.25, 0.75)


    # Random storm intensity
    intensity = rng.uniform(30, 60)


    # Extreme weather feature
    storm = intensity * np.exp(
        -(
            (X - center_x) ** 2
            + (Y - center_y) ** 2
        ) / 0.015
    )


    weather += storm


    # Add small-scale variation
    noise = rng.normal(
        0,
        0.8,
        (fine_size, fine_size)
    )

    weather += noise


    # ------------------------------------
    # Fine-resolution target
    # ------------------------------------

    fine_targets.append(
        weather.astype(np.float32)
    )


    # ------------------------------------
    # Create coarse-resolution input
    # ------------------------------------

    coarse = weather.reshape(
        coarse_size,
        fine_size // coarse_size,
        coarse_size,
        fine_size // coarse_size
    ).mean(axis=(1, 3))


    coarse_inputs.append(
        coarse.astype(np.float32)
    )


# ----------------------------------------
# Convert to PyTorch tensors
# ----------------------------------------

X = torch.tensor(
    np.array(coarse_inputs),
    dtype=torch.float32
).unsqueeze(1)


Y = torch.tensor(
    np.array(fine_targets),
    dtype=torch.float32
).unsqueeze(1)


# ----------------------------------------
# Save dataset
# ----------------------------------------

torch.save(
    {
        "inputs": X,
        "targets": Y
    },
    output_file
)


# ----------------------------------------
# Display information
# ----------------------------------------

print("AI Downscaling Dataset Created")
print("=" * 45)

print()
print("Dataset file:", output_file)

print()
print("Number of samples:", num_samples)

print()
print("Input shape:")
print(X.shape)

print()
print("Target shape:")
print(Y.shape)

print()
print("Input resolution:", coarse_size, "x", coarse_size)
print("Target resolution:", fine_size, "x", fine_size)

print()
print("Dataset saved successfully!")