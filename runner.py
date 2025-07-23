import os
import argparse

def run_main():
    print("🔄 Running main detection script...")
    os.system("python main.py")

def run_interpolation():
    print("🔄 Running interpolation to fill missing data...")
    os.system("python add_missing_data.py")

def run_visualization():
    print("🔄 Running visualization to generate final video...")
    os.system("python visualize.py")

def main():
    parser = argparse.ArgumentParser(
        description="🔍 Number Plate Detection CLI using YOLOv8 and OpenCV")
    parser.add_argument(
        "--run", 
        choices=["all", "detect", "interpolate", "visualize"], 
        required=True,
        help="Stage to run: 'detect', 'interpolate', 'visualize', or 'all'"
    )
    args = parser.parse_args()

    if args.run == "all":
        run_main()
        run_interpolation()
        run_visualization()
    elif args.run == "detect":
        run_main()
    elif args.run == "interpolate":
        run_interpolation()
    elif args.run == "visualize":
        run_visualization()

if __name__ == "__main__":
    main()
