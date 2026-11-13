"""Run after installing paper-layer-counter; works from any directory."""
import argparse
import json

from paper_counter import PaperDetector


def main():
    parser = argparse.ArgumentParser(description="Paper counting API example")
    parser.add_argument("--image", required=True)
    parser.add_argument("--rotate-ccw", action="store_true")
    parser.add_argument("--preset", default="paper_layers_v4", choices=["tracks", "tracks_preprocessed", "regions", "paper_layers_v4", "e3_3"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = PaperDetector(preset=args.preset).detect(
        args.image, rotate_ccw=args.rotate_ccw, output_dir=args.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
