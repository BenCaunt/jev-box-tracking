"""Client for asking TypeSafe Jev where an object is in a frame."""

import base64
import os
import uuid

import cv2
import requests

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-preview"
SEND_WIDTH = 640  # Downscale frames before upload to cut latency.


class JevClient:
    """
    Asks Jev two questions about a frame: is `object_name` present, and where is its box.

    The API key defaults to the TYPESAFE_API_KEY environment variable.
    """

    def __init__(self, object_name, api_key=None, api_url=API_URL, model=MODEL,
                 send_width=SEND_WIDTH, timeout=30):
        self.object_name = object_name
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key: pass api_key or set TYPESAFE_API_KEY.")
        self.api_url = api_url
        self.model = model
        self.send_width = send_width
        self.timeout = timeout
        self.session = requests.Session()

    def detect(self, frame):
        """Send a BGR frame and return Jev's raw answers dict, or None if encoding fails."""
        h, w = frame.shape[:2]
        if w > self.send_width:
            frame = cv2.resize(frame, (self.send_width, round(h * self.send_width / w)))
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        ref = str(uuid.uuid4())
        body = {
            "model": self.model,
            "state": {"frame": ref},
            "assets": {
                ref: {"type": "image", "format": "jpg", "data_base64": base64.b64encode(jpg).decode()}
            },
            "questions": {
                "present": {"type": "noul", "instructions": f"Is there a {self.object_name} in `frame`?"},
                self.object_name: {"type": "box", "instructions": f"the {self.object_name} in `frame`"},
            },
        }
        resp = self.session.post(
            self.api_url,
            json=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "TypeSafe-Danger-Zone-Images": "true",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["answers"]
