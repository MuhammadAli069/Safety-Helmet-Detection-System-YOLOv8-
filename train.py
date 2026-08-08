import argparse
from pathlib import Path

from ultralytics import YOLO


def maybe_mount_drive(enable: bool) -> None:
    """Mount Google Drive when running in Colab."""
    if not enable:
        return

    try:
        from google.colab import drive  # type: ignore

        drive.mount('/content/drive', force_remount=False)
        print('Google Drive mounted at /content/drive')
    except Exception as exc:
        print(f'Skipping Google Drive mount: {exc}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train YOLOv8 for helmet detection.')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--project', type=str, default='outputs/training_results', help='Output directory')
    parser.add_argument('--name', type=str, default='yolov8n_helmet', help='Run name')
    parser.add_argument('--mount-drive', action='store_true', help='Mount Google Drive in Colab')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_mount_drive(args.mount_drive)

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO('yolov8n.pt')
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
    )

    print('Training complete.')
    print(f'Results saved under: {project_dir / args.name}')
    print(results)


if __name__ == '__main__':
    main()
