"""
VisionGuard AI — Real-time Object Detector
Author: Shebin S Illikkal
"""
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import List, Dict, Tuple
import threading
import queue


class SecurityDetector:
    CLASSES = ['person', 'fire', 'smoke', 'vehicle', 'unknown']
    ALERT_THRESHOLD = 0.72

    def __init__(self, model_path: str, alert_callback=None):
        print("Loading detection model...")
        self.model = tf.saved_model.load(model_path)
        self.infer = self.model.signatures['serving_default']
        self.alert_callback = alert_callback
        self.alert_queue = queue.Queue()
        self.frame_count = 0
        print("Model loaded.")

    def preprocess(self, frame: np.ndarray) -> tf.Tensor:
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        return tf.expand_dims(tensor, 0)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        input_tensor = self.preprocess(frame)
        output = self.infer(input_tensor=input_tensor)
        boxes   = output['detection_boxes'].numpy()[0]
        scores  = output['detection_scores'].numpy()[0]
        classes = output['detection_classes'].numpy()[0].astype(int)
        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            if score >= self.ALERT_THRESHOLD:
                label = self.CLASSES[cls] if cls < len(self.CLASSES) else 'unknown'
                detections.append({
                    'label': label, 'confidence': float(score),
                    'box': box.tolist(), 'timestamp': datetime.now().isoformat()
                })
        return detections

    def process_stream(self, camera_url: str, camera_id: str):
        cap = cv2.VideoCapture(camera_url)
        print(f"Stream started: {camera_id}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            if self.frame_count % 3 == 0:  # Process every 3rd frame
                detections = self.detect(frame)
                for det in detections:
                    if det['label'] in ('fire', 'smoke') or \
                       (det['label'] == 'person' and self._in_restricted_zone(det['box'])):
                        self._trigger_alert(camera_id, det, frame)
        cap.release()

    def _in_restricted_zone(self, box: list) -> bool:
        # Define restricted zones per camera — override in subclass
        return False

    def _trigger_alert(self, camera_id: str, detection: Dict, frame: np.ndarray):
        alert = {'camera': camera_id, 'detection': detection,
                 'frame_path': self._save_frame(frame)}
        self.alert_queue.put(alert)
        if self.alert_callback:
            threading.Thread(target=self.alert_callback, args=(alert,), daemon=True).start()

    def _save_frame(self, frame: np.ndarray) -> str:
        path = f"alerts/frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(path, frame)
        return path
