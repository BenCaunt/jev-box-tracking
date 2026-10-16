"""Realtime object tracking from TypeSafe Jev Box questions.

Jev gives an accurate box but slowly (one answer per API round trip). Between
answers, sparse optical flow (Lucas-Kanade + RANSAC similarity fit) on features
inside the box measures how the object moves frame to frame. A Kalman filter
fuses both: flow updates velocity every frame, Jev updates absolute position.

Jev answers are about frames from the past. Every frame and the filter state
after it are kept in a history buffer; when an answer arrives for frame k, the
filter is rewound to frame k, corrected with the Jev box, then re-run over the
saved frames k+1..now (faster than real time) to get the current estimate.
"""

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .client import JevClient
from .filter import FrameRecord, Track, replay

TRACK_WIDTH = 480  # Downscale frames for optical flow.
HISTORY_SECONDS = 5.0  # Jev answers older than this can't be replayed and are dropped.
MIN_PRESENT = 0.5  # Hide the box when the "is there an object" answer is below this.
MIN_CONFIDENCE = 0.7


class Detector(threading.Thread):
    """Background thread that keeps asking Jev about the most recently submitted frame."""

    def __init__(self, client):
        super().__init__(daemon=True)
        self.client = client
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.answers = None
        self.answers_frame_id = 0
        self.seq = 0
        self.latency = 0.0
        self.error = None
        self.running = True

    def submit(self, frame, frame_id):
        with self.lock:
            self.frame = frame
            self.frame_id = frame_id

    def run(self):
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
                frame_id = self.frame_id
            if frame is None:
                time.sleep(0.01)
                continue
            start = time.time()
            try:
                answers = self.client.detect(frame)
                self.error = None
            except Exception as e:  # Keep streaming through transient API errors.
                self.error = str(e)[:120]
                continue
            with self.lock:
                self.answers = answers
                self.answers_frame_id = frame_id
                self.seq += 1
                self.latency = time.time() - start


@dataclass
class TrackResult:
    """The tracker's estimate after a frame. Box coordinates are image fractions."""

    box: Optional[Tuple[float, float, float, float]]  # (left, top, right, bottom), or None if not tracking.
    points: Optional[np.ndarray]  # Tracked features in pixels of the input frame, shape (N, 2).
    present: float  # Latest Jev "is the object there" score.
    confidence: float  # Latest Jev box confidence.
    latency: float  # Seconds the latest Jev round trip took.
    replay_ms: float  # Milliseconds the latest rewind-and-replay took.
    answers: Optional[dict]  # Latest raw Jev answers.
    error: Optional[str]  # Latest Jev request error, if the last request failed.

    def pixel_box(self, width, height):
        """The box as integer pixel corners ((x1, y1), (x2, y2)), or None."""
        if self.box is None:
            return None
        left, top, right, bottom = self.box
        return (round(left * width), round(top * height)), (round(right * width), round(bottom * height))


class JevTracker:
    """
    Track `object_name` across frames: call `update(frame)` with every BGR frame.

    Jev requests run on a background thread; `update` never blocks on the network.
    Use as a context manager, or call `close()` when done.
    """

    def __init__(self, object_name, api_key=None, client=None, track_width=TRACK_WIDTH,
                 history_seconds=HISTORY_SECONDS, min_present=MIN_PRESENT,
                 min_confidence=MIN_CONFIDENCE):
        self.object_name = object_name
        self.client = client or JevClient(object_name, api_key=api_key)
        self.track_width = track_width
        self.history_seconds = history_seconds
        self.min_present = min_present
        self.min_confidence = min_confidence
        self.detector = Detector(self.client)
        self.detector.start()
        self.history = deque()
        self.track = None
        self.frame_id = 0
        self.last_seq = 0
        self.present = self.confidence = 0.0
        self.replay_ms = 0.0

    def update(self, frame, t=None):
        """Process one BGR frame (captured at time `t`, default now) and return a TrackResult."""
        t = time.time() if t is None else t
        self.frame_id += 1
        self.detector.submit(frame.copy(), self.frame_id)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(
            cv2.resize(frame, (self.track_width, round(h * self.track_width / w))), cv2.COLOR_BGR2GRAY)

        # Track forward one frame with optical flow.
        history = self.history
        if self.track is not None and history:
            self.track.step(history[-1].gray, gray, max(t - history[-1].t, 1e-3))
        history.append(FrameRecord(self.frame_id, t, gray, copy.deepcopy(self.track)))
        while history and t - history[0].t > self.history_seconds:
            history.popleft()

        with self.detector.lock:
            answers, latency = self.detector.answers, self.detector.latency
            seq, answer_frame_id = self.detector.seq, self.detector.answers_frame_id

        # A new Jev answer: rewind to the frame it was asked about, correct, replay to now.
        if answers and seq != self.last_seq:
            self.last_seq = seq
            self.present = float(answers.get("present", {}).get("noul", 1.0))
            obj = answers.get(self.object_name)
            self.confidence = obj.get("confidence", 0.0) if obj else 0.0
            start = answer_frame_id - history[0].frame_id
            if 0 <= start < len(history):
                replay_start = time.time()
                record = history[start]
                past = copy.deepcopy(record.track)
                if self.present >= self.min_present and self.confidence >= self.min_confidence:
                    b = obj["box"]
                    centre = np.array([(b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2])
                    size = (b["right"] - b["left"], b["bottom"] - b["top"])
                    if past is None:
                        past = Track(centre, size, record.gray)
                    else:
                        past.correct(centre, size, record.gray)
                else:
                    past = None
                self.track = replay(history, start, past)
                self.replay_ms = (time.time() - replay_start) * 1000

        points = None
        if self.track is not None and self.track.points is not None:
            points = self.track.points.reshape(-1, 2) * (w / self.track_width)
        return TrackResult(
            box=tuple(float(v) for v in self.track.box()) if self.track is not None else None,
            points=points,
            present=self.present,
            confidence=self.confidence,
            latency=latency,
            replay_ms=self.replay_ms,
            answers=answers,
            error=self.detector.error,
        )

    def close(self):
        self.detector.running = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
