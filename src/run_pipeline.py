import os
import sys
import subprocess


# ============================================================
# EXTREMEWEATHERAI - MASTER PIPELINE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)


def run_step(title, script):
    """Run one pipeline script."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    script_path = os.path.join(
        SRC_DIR,
        script
    )

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found:")
        print(script_path)
        return False

    try:
        subprocess.run(
            [
                sys.executable,
                script_path
            ],
            cwd=PROJECT_ROOT,
            check=True
        )

        print()
        print(f"✓ {title} completed successfully")

        return True

    except subprocess.CalledProcessError as error:

        print()
        print(f"✗ {title} failed")
        print(f"Exit code: {error.returncode}")

        return False


def main():

    print()
    print("=" * 70)
    print("EXTREMEWEATHERAI")
    print("MASTER AI WEATHER PIPELINE")
    print("=" * 70)

    print()
    print("Project:")
    print(PROJECT_ROOT)

    print()


    # ========================================================
    # PIPELINE STEPS
    # ========================================================

    steps = [

        (
            "1. Creating Weather Dataset",
            os.path.join(
                "data",
                "create_sample_weather.py"
            )
        ),

        (
            "2. Detecting Extreme Weather",
            os.path.join(
                "data",
                "detect_extreme.py"
            )
        ),

        (
            "3. Detecting Anomaly Region",
            os.path.join(
                "data",
                "detect_anomaly_region.py"
            )
        ),

        (
            "4. Creating Dynamic Bounding Boxes",
            os.path.join(
                "data",
                "create_bounding_boxes.py"
            )
        ),

        (
            "5. Creating Weather Event Records",
            os.path.join(
                "data",
                "create_event_records.py"
            )
        ),

        (
            "6. Creating Trajectory Dataset",
            os.path.join(
                "data",
                "create_trajectory_dataset.py"
            )
        ),

        (
            "7. Building Spatio-Temporal Graph",
            os.path.join(
                "data",
                "build_spatiotemporal_graph.py"
            )
        ),

        (
            "8. Predicting Weather Trajectory",
            os.path.join(
                "data",
                "predict_trajectory.py"
            )
        ),

        (
            "9. Classifying Weather Risk",
            os.path.join(
                "alerts",
                "classify_risk.py"
            )
        ),

        (
            "10. Creating Emergency Alert",
            os.path.join(
                "alerts",
                "create_emergency_alert.py"
            )
        )

    ]


    # ========================================================
    # RUN PIPELINE
    # ========================================================

    successful = 0

    for title, script in steps:

        success = run_step(
            title,
            script
        )

        if not success:

            print()
            print("=" * 70)
            print("PIPELINE STOPPED")
            print("=" * 70)

            print()
            print("Failed step:")
            print(title)

            print()
            print("Fix this step before continuing.")

            return

        successful += 1


    # ========================================================
    # FINAL STATUS
    # ========================================================

    total = len(steps)

    percentage = (
        successful / total
    ) * 100


    print()
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    print()

    print(
        f"Completed: {successful}/{total}"
    )

    print(
        f"Completion: {percentage:.0f}%"
    )

    print()

    print("Generated outputs should now be available in:")

    print(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "processed"
        )
    )

    print()

    print("=" * 70)


if __name__ == "__main__":
    main()