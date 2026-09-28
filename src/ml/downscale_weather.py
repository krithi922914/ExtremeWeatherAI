import numpy as np
import matplotlib.pyplot as plt


print("12 km -> 5 km Weather Downscaling")
print("=" * 45)


# ----------------------------------------
# Create a synthetic coarse 12 km grid
# ----------------------------------------

coarse_size = 10

x_coarse = np.linspace(0, 1, coarse_size)
y_coarse = np.linspace(0, 1, coarse_size)

X_coarse, Y_coarse = np.meshgrid(
    x_coarse,
    y_coarse
)


# Synthetic weather field
weather_coarse = (
    20
    + 10 * X_coarse
    + 5 * Y_coarse
)


# Add an extreme weather feature
storm = 40 * np.exp(
    -(
        (X_coarse - 0.65) ** 2
        + (Y_coarse - 0.55) ** 2
    ) / 0.02
)

weather_coarse += storm


# ----------------------------------------
# Create higher-resolution 5 km grid
# ----------------------------------------

fine_size = 24

x_fine = np.linspace(0, 1, fine_size)
y_fine = np.linspace(0, 1, fine_size)


# ----------------------------------------
# Interpolate along longitude
# ----------------------------------------

intermediate = np.zeros(
    (coarse_size, fine_size)
)

for i in range(coarse_size):

    intermediate[i] = np.interp(
        x_fine,
        x_coarse,
        weather_coarse[i]
    )


# ----------------------------------------
# Interpolate along latitude
# ----------------------------------------

weather_fine = np.zeros(
    (fine_size, fine_size)
)

for j in range(fine_size):

    weather_fine[:, j] = np.interp(
        y_fine,
        y_coarse,
        intermediate[:, j]
    )


# ----------------------------------------
# Display information
# ----------------------------------------

print()
print("Coarse 12 km grid:")
print("Shape:", weather_coarse.shape)

print()

print("Downscaled 5 km grid:")
print("Shape:", weather_fine.shape)

print()

print(
    "Maximum coarse value:",
    f"{weather_coarse.max():.2f}"
)

print(
    "Maximum fine value:",
    f"{weather_fine.max():.2f}"
)


# ----------------------------------------
# Visualize both grids
# ----------------------------------------

plt.figure(figsize=(10, 7))

plt.imshow(
    weather_coarse,
    origin="lower",
    extent=[0, 1, 0, 1],
    aspect="auto"
)

plt.colorbar(label="Weather Intensity")

plt.xlabel("Normalized Longitude")
plt.ylabel("Normalized Latitude")

plt.title("Coarse 12 km Weather Field")

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 7))

plt.imshow(
    weather_fine,
    origin="lower",
    extent=[0, 1, 0, 1],
    aspect="auto"
)

plt.colorbar(label="Weather Intensity")

plt.xlabel("Normalized Longitude")
plt.ylabel("Normalized Latitude")

plt.title("Downscaled 5 km Weather Field")

plt.tight_layout()
plt.show()