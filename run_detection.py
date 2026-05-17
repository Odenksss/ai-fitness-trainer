import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Exercise Pose Correction")
    parser.add_argument(
        "--exercise",
        choices=["bicep", "plank", "squat", "lunge"],
        required=True,
        help="Which exercise to analyze"
    )
    parser.add_argument(
        "--source",
        default=0,
        help="Video source: 0 for webcam, or path to video file"
    )
    args = parser.parse_args()

    # Convert source to int if it's a digit string
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print(f"\n=== Exercise Correction: {args.exercise.upper()} ===")
    print("Press 'q' to quit\n")

    if args.exercise == "bicep":
        from core.bicep_model.bicep_detector import run_bicep_curl_detection
        run_bicep_curl_detection(source)

    elif args.exercise == "plank":
        from core.plank_model.plank_detector import run_plank_detection
        run_plank_detection(source)

    elif args.exercise == "squat":
        from core.squat_model.squat_detector import run_squat_detection
        run_squat_detection(source)

    elif args.exercise == "lunge":
        from core.lunge_model.lunge_detector import run_lunge_detection
        run_lunge_detection(source)


if __name__ == "__main__":
    main()