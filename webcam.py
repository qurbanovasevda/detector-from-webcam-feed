from pathlib import Path
from collections import deque
from datetime import datetime
import time

import cv2
from ultralytics import YOLO


model_path = Path(__file__).parent / "pen_detector" / "best.pt"

if not model_path.exists():
    raise FileNotFoundError(f"Model tapılmadı: {model_path}")

model = YOLO(str(model_path))

# Çıxış qovluqları (screenshot və video)
screenshots_dir = Path(__file__).parent / "screenshots"
recordings_dir = Path(__file__).parent / "recordings"
screenshots_dir.mkdir(exist_ok=True)
recordings_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Kamera açılmadı. Kamera indeksini 0-dan 1-ə dəyiş.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

window_name = "Pen Detector"
cv2.namedWindow(window_name)


def on_threshold_change(value):
    pass


cv2.createTrackbar(
    "Confidence %",
    window_name,
    50,
    100,
    on_threshold_change
)

ok, frame = cap.read()

if not ok:
    cap.release()
    raise RuntimeError("Kameradan görüntü alınmadı.")

model.predict(frame, imgsz=640, device="cpu", verbose=False)

frame_times = deque(maxlen=30)

# Video yazma vəziyyəti
writer = None          # cv2.VideoWriter obyekti (yazılış zamanı doludur)
recording_path = None


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def start_recording(frame_size, fps):
    fps = min(max(fps, 5.0), 30.0)
    path = recordings_dir / f"rec_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not vw.isOpened():
        print("Video faylı açıla bilmədi.")
        return None, None
    print(f"Video yazılışı başladı: {path} ({fps:.1f} FPS)")
    return vw, path


def stop_recording(vw, path):
    vw.release()
    print(f"Video saxlandı: {path}")


try:
    while True:
        start = time.perf_counter()

        ok, frame = cap.read()

        if not ok:
            print("Kameradan görüntü alınmadı.")
            break

        threshold = cv2.getTrackbarPos(
            "Confidence %",
            window_name
        ) / 100

        threshold = max(0.01, threshold)

        result = model.predict(
            source=frame,
            imgsz=640,
            conf=threshold,
            iou=0.45,
            device="cpu",
            verbose=False
        )[0]

        output = result.plot()

        cv2.putText(
            output,
            f"Confidence threshold: {threshold:.2f}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Detections: {len(result.boxes)} | S: screenshot | R: record | Q: exit",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        # Son tamamlanan kadrların orta FPS-i.
        fps = len(frame_times) / sum(frame_times) if frame_times else 0

        cv2.putText(
            output,
            f"FPS: {fps:.1f} | CPU",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Yazılış göstəricisi (qırmızı nöqtə + REC)
        if writer is not None:
            h, w = output.shape[:2]
            cv2.circle(output, (w - 70, 25), 8, (0, 0, 255), -1)
            cv2.putText(
                output,
                "REC",
                (w - 55, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            # Videoya yazılır: boxlar, FPS və threshold
            writer.write(output)

        cv2.imshow(window_name, output)
        key = cv2.waitKey(1) & 0xFF

        # Kamera oxuma + inference + drawing + display müddəti.
        frame_times.append(time.perf_counter() - start)

        if key == ord("q"):
            break

        elif key == ord("s"):
            # Fayl adında threshold və aşkarlanma sayı
            name = f"shot_{timestamp()}_conf{int(threshold * 100)}_det{len(result.boxes)}.png"
            path = screenshots_dir / name
            cv2.imwrite(str(path), output)
            print(f"Screenshot saxlandı: {path}")

        elif key == ord("r"):
            if writer is None:
                h, w = output.shape[:2]
                writer, recording_path = start_recording((w, h), fps if fps > 0 else 15.0)
            else:
                stop_recording(writer, recording_path)
                writer = None
                recording_path = None

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    if writer is not None:
        stop_recording(writer, recording_path)
    cap.release()
    cv2.destroyAllWindows()