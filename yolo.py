from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")  # 预训练模型
    video_path = Path(__file__).with_name("屏幕录制 2026-10-02 122538.mp4")

    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    output_dir = Path(__file__).parent / "output"
    results = model.track(
        source=str(video_path),
        conf=0.25,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        save=True,
        show_labels=True,
        project=str(Path(__file__).parent),
        name="output",
        exist_ok=True,
    )
    for result in results:
        if result.boxes.id is not None:
            print(f"帧 {result.path}: track IDs {result.boxes.id.int().cpu().tolist()}")
    print(f"检测视频已保存到: {output_dir}")

    # 可选：输出检测结果


if __name__ == "__main__":
    main()