"""Optical flow tracking fused with Jev boxes through a Kalman filter."""

import copy

import cv2
import numpy as np

JEV_NOISE = 2e-4  # Variance of a Jev box centre, in image fractions squared.
FLOW_NOISE = 1e-3  # Variance of a flow velocity measurement, in (fractions / s) squared.
PROCESS_NOISE = 0.5  # Acceleration noise of the constant velocity model.
MIN_POINTS = 8  # Below this many tracked features, flow is skipped and features reseeded.


class ConstantVelocityKalmanFilter:
    """
    Kalman filter on the box centre with a constant velocity model.

    State is [x, y, vx, vy] in image fractions. Jev measures position, optical
    flow measures velocity.
    """

    H_POSITION = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    H_VELOCITY = np.array([[0, 0, 1.0, 0], [0, 0, 0, 1.0]])

    def __init__(self, position, process_noise=PROCESS_NOISE):
        self.x = np.zeros(4)
        self.x[:2] = position
        self.P = np.diag([JEV_NOISE, JEV_NOISE, 0.1, 0.1])
        self.q = process_noise

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # Noise enters through the acceleration, integrated up to velocity/position.
        g = np.array([dt**2 / 2, dt**2 / 2, dt, dt])
        Q = np.outer(g, g) * self.q * np.kron(np.ones((2, 2)), np.eye(2))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, variance):
        y = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(2) * variance
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


class Track:
    """Filter state plus the box size and the features being followed inside it."""

    def __init__(self, centre, size, gray):
        self.kf = ConstantVelocityKalmanFilter(centre)
        self.size = np.array(size, dtype=float)
        self.points = seed_points(gray, centre, self.size)

    def correct(self, centre, size, gray):
        """Apply a Jev box measured on `gray`, and restart features from it."""
        self.kf.update(centre, ConstantVelocityKalmanFilter.H_POSITION, JEV_NOISE)
        self.size = np.array(size, dtype=float)
        self.points = seed_points(gray, self.kf.x[:2], self.size)

    def step(self, prev_gray, gray, dt):
        """Advance one frame: predict, then update velocity from optical flow."""
        h, w = gray.shape
        prev_centre = self.kf.x[:2].copy()
        self.kf.predict(dt)
        if self.points is not None and len(self.points) >= MIN_POINTS:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, self.points, None, winSize=(21, 21), maxLevel=3)
            good = status.ravel() == 1
            p0, p1 = self.points[good], nxt[good]
            M, inliers = (None, None)
            if len(p0) >= MIN_POINTS:
                M, inliers = cv2.estimateAffinePartial2D(
                    p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is not None and inliers.sum() >= MIN_POINTS:
                # Where the similarity transform moves the old box centre.
                c0 = prev_centre * (w, h)
                c1 = M[:, :2] @ c0 + M[:, 2]
                velocity = (c1 - c0) / (w, h) / dt
                self.kf.update(velocity, ConstantVelocityKalmanFilter.H_VELOCITY, FLOW_NOISE)
                self.size *= np.sqrt(abs(np.linalg.det(M[:, :2])))
                self.points = p1[inliers.ravel() == 1].reshape(-1, 1, 2)
            else:
                self.points = None
        if self.points is None or len(self.points) < MIN_POINTS:
            self.points = seed_points(gray, self.kf.x[:2], self.size)

    def box(self):
        """(left, top, right, bottom) in image fractions."""
        cx, cy = self.kf.x[:2]
        bw, bh = self.size
        return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def seed_points(gray, centre, size):
    h, w = gray.shape
    left, top = (np.array(centre) - np.array(size) / 2) * (w, h)
    right, bottom = (np.array(centre) + np.array(size) / 2) * (w, h)
    mask = np.zeros_like(gray)
    mask[max(0, int(top)):max(0, int(bottom)), max(0, int(left)):max(0, int(right))] = 255
    return cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=5, mask=mask)


class FrameRecord:
    def __init__(self, frame_id, t, gray, track):
        self.frame_id = frame_id
        self.t = t
        self.gray = gray
        self.track = track  # Snapshot of the track after processing this frame, or None.


def replay(history, start, track):
    """Re-run `track` from history[start] to the newest frame, rewriting snapshots."""
    history[start].track = copy.deepcopy(track)
    for i in range(start + 1, len(history)):
        prev, cur = history[i - 1], history[i]
        if track is not None:
            track.step(prev.gray, cur.gray, max(cur.t - prev.t, 1e-3))
        cur.track = copy.deepcopy(track)
    return track
