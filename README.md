# jev-box-tracking

Realtime object tracking using [TypeSafe](https://typesafe.ai) Jev Box questions.

Jev gives an accurate box but slowly (one answer per API round trip). Between answers, sparse optical flow (Lucas-Kanade + RANSAC similarity fit) on features inside the box measures how the object moves frame to frame. A Kalman filter fuses both: flow updates velocity every frame, Jev updates absolute position.

Jev answers are about frames from the past, so every frame and the filter state after it are kept in a history buffer. When an answer arrives for frame k, the filter is rewound to frame k, corrected with the Jev box, then re-run over frames k+1..now to get the current estimate.

## Install

```sh
pip install "git+https://github.com/BenCaunt/jev-box-tracking.git"
```

Set your API key:

```sh
export TYPESAFE_API_KEY=...
```

## Webcam demo

```sh
jev-box-track "iPhone"          # press q to quit
jev-box-track "coffee mug" --camera 1
```

Install with `[dotenv]` to have the demo load `TYPESAFE_API_KEY` from a `.env` file:

```sh
pip install "jev-box-tracking[dotenv] @ git+https://github.com/BenCaunt/jev-box-tracking.git"
```

## Library

```python
import cv2
from jev_box_tracking import JevTracker

cap = cv2.VideoCapture(0)
with JevTracker("iPhone") as tracker:  # api_key=... or TYPESAFE_API_KEY
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = tracker.update(frame)  # never blocks on the network
        if result.box is not None:
            left, top, right, bottom = result.box  # image fractions, 0..1
            print(result.pixel_box(frame.shape[1], frame.shape[0]))
```

`TrackResult` also carries `present`, `confidence`, `latency`, `replay_ms`, the tracked feature `points`, the raw Jev `answers`, and any request `error`.

For a single synchronous request, use `JevClient("iPhone").detect(frame)`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
