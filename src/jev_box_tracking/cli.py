"""Webcam demo: `jev-box-track "iPhone"`. Press q to quit."""

import argparse
import time

import cv2

from .tracker import JevTracker

COLOR = (186, 81, 229)


def open_camera(index):
    # On macOS the first open triggers the camera permission prompt asynchronously
    # and fails immediately, so keep retrying while the user answers it.
    cap = cv2.VideoCapture(index)
    deadline = time.time() + 30
    while not cap.isOpened() and time.time() < deadline:
        time.sleep(1)
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (check camera permissions for your terminal).")
    return cap


def draw(frame, result, object_name):
    h, w = frame.shape[:2]
    corners = result.pixel_box(w, h)
    if corners is not None:
        p1, p2 = corners
        cv2.rectangle(frame, p1, p2, COLOR, 3)
        cv2.putText(frame, f"{object_name} {result.confidence:.2f}", (p1[0], max(20, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
    if result.points is not None:
        for x, y in result.points:
            cv2.circle(frame, (round(x), round(y)), 2, (0, 255, 0), -1)
    if result.answers:
        cv2.putText(frame, f"present {result.present:.2f}  latency {result.latency * 1000:.0f}ms  "
                    f"replay {result.replay_ms:.0f}ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if result.error:
        cv2.putText(frame, result.error, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Track an object on the webcam with TypeSafe Jev.")
    parser.add_argument("object", nargs="?", default="iPhone", help="What to track (default: iPhone).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    cap = open_camera(args.camera)
    with JevTracker(args.object) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = tracker.update(frame)
            draw(frame, result, args.object)
            cv2.imshow(f"Jev Box - {args.object}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
