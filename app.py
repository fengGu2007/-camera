"""Local web monitor for animal, smoke and fire detection."""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_file
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ANIMAL_MODEL_PATH = Path(os.getenv("ANIMAL_MODEL", ROOT / "yolov8n.pt"))
FIRE_MODEL_PATH = Path(os.getenv("FIRE_MODEL", ROOT / "yolo11-d-fire-dataset.pt"))
CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
ALARM_AFTER_SECONDS = 1.0
ANIMAL_LABELS = {
    "cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "猫", "狗", "小狗", "犬", "鸟", "马", "羊", "牛", "大象", "熊", "斑马", "长颈鹿",
    "兔子", "兔", "猪", "鹿", "狐狸", "狼", "猴子", "鸡", "鸭", "宠物",
}
app = Flask(__name__)


class Monitor:
    def __init__(self):
        self.lock = threading.RLock()
        self.animal_model = None
        self.fire_model = None
        self.running = False
        self.frame = None
        self.events = []
        self.status = "等待上传视频"
        self.source_name = ""
        self.started_at = 0.0

    def ensure_models(self):
        with self.lock:
            if self.animal_model is None:
                if not ANIMAL_MODEL_PATH.exists():
                    raise FileNotFoundError(f"找不到动物模型：{ANIMAL_MODEL_PATH}")
                self.animal_model = YOLO(str(ANIMAL_MODEL_PATH))
            if self.fire_model is None:
                if not FIRE_MODEL_PATH.exists():
                    raise FileNotFoundError(f"找不到火烟模型：{FIRE_MODEL_PATH}")
                self.fire_model = YOLO(str(FIRE_MODEL_PATH))

    def start(self, path: Path, display_name: str):
        with self.lock:
            if self.running:
                raise RuntimeError("当前已有视频正在检测，请先停止")
            self.events = []
            self.frame = None
            self.status = "正在加载模型…"
            self.source_name = display_name
            self.running = True
        threading.Thread(target=self.process, args=(path,), daemon=True).start()

    def process(self, path: Path):
        active = {}  # alert category -> (first seen, last seen, fired)
        cap = None
        try:
            self.ensure_models()
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError("无法打开视频文件")
            with self.lock:
                self.status = "检测中"
                self.started_at = time.time()
            while True:
                with self.lock:
                    if not self.running:
                        break
                ok, image = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                with self.lock:
                    animal_model, fire_model = self.animal_model, self.fire_model
                results = [(animal_model, "animal"), (fire_model, "hazard")]
                categories_seen = set()
                for model, kind in results:
                    result = model.predict(image, conf=CONFIDENCE, verbose=False)[0]
                    names = result.names
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        label = str(names[class_id])
                        normalized = label.lower().replace("_", " ").strip()
                        if kind == "animal":
                            # COCO animal classes plus common custom animal labels.
                            is_animal = normalized in ANIMAL_LABELS or any(
                                word in normalized for word in ("animal", "cat", "dog", "pet", "wildlife", "猫", "狗", "犬", "动物")
                            )
                            if not is_animal:
                                continue
                            category = "动物"
                            alert_label = label
                            color = (40, 210, 80)
                        else:
                            if not any(word in normalized for word in ("fire", "flame", "smoke")):
                                continue
                            category = "火情" if any(word in normalized for word in ("fire", "flame")) else "烟雾"
                            alert_label = label
                            color = (30, 60, 245)
                        categories_seen.add(category)
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0].item())
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        caption = f"{label} {confidence:.0%}"
                        cv2.putText(image, caption, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                for category in categories_seen:
                    first, last, fired = active.get(category, (now, now, False))
                    if now - last > 0.6:
                        first, fired = now, False
                    last = now
                    if not fired and now - first >= ALARM_AFTER_SECONDS:
                        active[category] = (first, last, True)
                        with self.lock:
                            self.events.insert(0, {"id": str(uuid.uuid4()), "category": category,
                                                  "time": time.strftime("%H:%M:%S"), "message": f"持续检测到{category}超过 1 秒"})
                            self.events = self.events[:30]
                    else:
                        active[category] = (first, last, fired)
                for category, (first, last, fired) in list(active.items()):
                    if category not in categories_seen and now - last > 0.6:
                        del active[category]
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    with self.lock:
                        self.frame = encoded.tobytes()
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                time.sleep(max(0, 1 / fps - 0.005))
            with self.lock:
                self.status = "视频播放完毕"
        except Exception as exc:
            with self.lock:
                self.status = f"检测失败：{exc}"
        finally:
            if cap:
                cap.release()
            with self.lock:
                self.running = False
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


monitor = Monitor()


@app.get("/")
def index():
    return send_file(ROOT / "index.html")


@app.post("/api/start")
def start():
    video = request.files.get("video")
    if not video or not video.filename:
        return jsonify(error="请选择视频文件"), 400
    suffix = Path(video.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return jsonify(error="支持 MP4、MOV、AVI、MKV、WebM 视频"), 400
    path = ROOT / "uploads" / f"{uuid.uuid4().hex}{suffix}"
    path.parent.mkdir(exist_ok=True)
    video.save(path)
    try:
        monitor.start(path, video.filename)
    except Exception as exc:
        path.unlink(missing_ok=True)
        return jsonify(error=str(exc)), 409
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    # The processing loop checks this flag between frames.
    with monitor.lock:
        monitor.running = False
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    with monitor.lock:
        return jsonify(status=monitor.status, running=monitor.running, source=monitor.source_name,
                       events=monitor.events, has_frame=monitor.frame is not None)


@app.get("/video")
def video_stream():
    def generate():
        last = None
        while True:
            with monitor.lock:
                frame, running = monitor.frame, monitor.running
            if frame and frame is not last:
                last = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            elif not running and frame is None:
                time.sleep(0.15)
            else:
                time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True, debug=False)
