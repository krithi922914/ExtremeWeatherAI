import os
import json


# ============================================================
# EXTREMEWEATHERAI
# DOWNSCALING RESULTS REPORT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "downscaling_results.json"
)


# ============================================================
# RESULTS FROM THE CURRENT PROTOTYPE EXPERIMENT
# ============================================================

results = {

    "status": "completed",

    "prototype": True,

    "input_resolution": "10 x 10",

    "output_resolution": "20 x 20",

    "dataset_samples": 200,

    "models": {

        "standard_cnn": {

            "name": "Standard CNN",

            "mse": 0.548185,

            "mae": 0.589521,

            "mean_max_error": 0.621413,

            "median_max_error": 0.523083,

            "extreme_pixel_mae": 0.638055,

            "description":
                "Baseline CNN downscaling model."
        },


        "extreme_cnn": {

            "name": "Extreme-Aware CNN",

            "mse": 0.698831,

            "mae": 0.661480,

            "mean_max_error": 0.521488,

            "median_max_error": 0.419609,

            "extreme_pixel_mae": 0.587120,

            "max_preservation_better_samples": 118,

            "max_preservation_percentage": 59.0,

            "description":
                "CNN trained with an extreme-aware loss "
                "to better preserve high-intensity weather regions."
        },


        "physics_extreme_cnn": {

            "name": "Physics + Extreme CNN",

            "mse": 0.717017,

            "mae": 0.672071,

            "mean_max_error": 0.563334,

            "median_max_error": 0.469994,

            "extreme_pixel_mae": 0.622835,

            "average_predicted_max": 70.120071,

            "description":
                "CNN combining extreme-aware loss with "
                "a simplified spatial-gradient physics penalty."
        }

    },


    "target_average_max": 70.361786,

    "standard_average_predicted_max": 69.977264,

    "extreme_average_predicted_max": 70.348633,

    "physics_extreme_average_predicted_max": 70.120071,


    "important_note":
        "These results are from the current synthetic "
        "prototype dataset and are not operational NWP "
        "validation results."
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print()
print("=" * 65)
print("DOWNSCALING RESULTS REPORT")
print("=" * 65)

print()

print("Models evaluated:")
print("  1. Standard CNN")
print("  2. Extreme-Aware CNN")
print("  3. Physics + Extreme CNN")

print()

print("Dataset samples:", results["dataset_samples"])

print(
    "Input resolution:",
    results["input_resolution"]
)

print(
    "Output resolution:",
    results["output_resolution"]
)

print()

print("Extreme-Aware CNN:")
print(
    "  MSE:",
    results["models"]["extreme_cnn"]["mse"]
)

print(
    "  MAE:",
    results["models"]["extreme_cnn"]["mae"]
)

print(
    "  Extreme-pixel MAE:",
    results["models"]["extreme_cnn"]["extreme_pixel_mae"]
)

print(
    "  Max preservation:",
    results["models"]["extreme_cnn"]
    ["max_preservation_percentage"],
    "%"
)

print()

print("Saved to:")
print(OUTPUT_FILE)

print()

print("=" * 65)
print("DOWNSCALING REPORT CREATED")
print("=" * 65)