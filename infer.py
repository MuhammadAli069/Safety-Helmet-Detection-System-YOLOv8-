import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on test images.')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights (.pt)')
    parser.add_argument('--source', type=str, default='dataset/images/test', help='Folder with test images')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
    parser.add_argument('--project', type=str, default='outputs/predictions', help='Output directory')
    parser.add_argument('--name', type=str, default='inference', help='Run name')
    parser.add_argument('--max-images', type=int, default=15, help='Max number of images to save')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f'Source folder not found: {source_dir}')

    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
        image_files.extend(source_dir.glob(ext))

    image_files = sorted(image_files)
    if not image_files:
        raise ValueError(f'No images found in {source_dir}')
    if len(image_files) < 15:
        raise ValueError(
            f'At least 15 test images are required, but found {len(image_files)} in {source_dir}'
        )

    selected = image_files[: max(args.max_images, 15)]

    model = YOLO(args.weights)
    model.predict(
        source=[str(p) for p in selected],
        conf=args.conf,
        save=True,
        save_txt=False,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    print(f'Inference complete. Saved predictions to: {Path(args.project) / args.name}')
    print(f'Images processed: {len(selected)}')


if __name__ == '__main__':
    main()
