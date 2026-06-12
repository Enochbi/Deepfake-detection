"""
video_processor.py - MULTI-FACE TRACKING
Extraction de frames avec tracking de visages
Support pour plusieurs visages simultanés dans la vidéo
"""

import os
import cv2
import numpy as np
import urllib.request
from collections import defaultdict


class VideoProcessor:
    """Extraction de frames et visages avec tracking multi-face"""

    def __init__(self, model_dir="models", min_confidence=0.5, iou_threshold=0.3):
        self.model_dir = model_dir
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.net = None
        self.face_size = (224, 224)
        os.makedirs(model_dir, exist_ok=True)

    def download_opencv_model(self):
        print("🔍 Vérification du détecteur OpenCV DNN...")

        prototxt_path = os.path.join(self.model_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(
            self.model_dir,
            "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )

        if not os.path.exists(prototxt_path):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                prototxt_path
            )

        if not os.path.exists(caffemodel_path):
            urls = [
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel",
                "https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            ]
            for url in urls:
                try:
                    urllib.request.urlretrieve(url, caffemodel_path)
                    break
                except Exception:
                    continue

        return prototxt_path, caffemodel_path

    def load_face_detector(self):
        proto, model = self.download_opencv_model()
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        return not self.net.empty()

    def detect_faces(self, frame):
        if self.net is None:
            return []

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.min_confidence:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                faces.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence
                })

        faces.sort(key=lambda f: f["confidence"], reverse=True)
        return faces

    def compute_iou(self, box1, box2):
        """Calcule l'Intersection over Union entre deux boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def associate_faces_to_tracks(self, faces, active_tracks):
        """
        Associe les visages détectés aux tracks existants via IOU
        Retourne: dict {track_id: face_info}
        """
        if not active_tracks:
            return {i: face for i, face in enumerate(faces)}
        
        assignments = {}
        used_faces = set()
        
        for track_id, last_box in active_tracks.items():
            best_iou = 0.0
            best_face_idx = None
            
            for i, face in enumerate(faces):
                if i in used_faces:
                    continue
                
                iou = self.compute_iou(last_box, face["box"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_face_idx = i
            
            if best_face_idx is not None:
                assignments[track_id] = faces[best_face_idx]
                used_faces.add(best_face_idx)
        
        next_track_id = max(active_tracks.keys()) + 1 if active_tracks else 0
        for i, face in enumerate(faces):
            if i not in used_faces:
                assignments[next_track_id] = face
                next_track_id += 1
        
        return assignments

    def extract_face_crop(self, frame, face_info, min_face_size=40):
        x1, y1, x2, y2 = face_info["box"]

        if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
            return None

        face = frame[y1:y2, x1:x2]
        if face is None or face.size == 0:
            return None

        return cv2.resize(face, self.face_size)

    def extract_frames_from_video(self, video_path, max_frames=10):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        indices = (
            list(range(total))
            if total <= max_frames
            else np.linspace(0, total - 1, max_frames, dtype=int)
        )

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append((int(idx), frame))

        cap.release()
        return frames

    def process_video(
        self,
        video_path,
        max_frames=10,
        max_faces_per_frame=5,
        save_faces=True,
        output_dir="static/faces"
    ):
        """
        Traitement vidéo avec tracking de visages
        
        Retourne:
        {
            "tracks": {
                track_id: [face_images],
                ...
            },
            "track_info": {
                track_id: {
                    "num_frames": int,
                    "first_appearance": int,
                    "last_appearance": int,
                    "avg_confidence": float
                }
            },
            "stats": {...}
        }
        """
        os.makedirs(output_dir, exist_ok=True)

        frames = self.extract_frames_from_video(video_path, max_frames)
        if not frames:
            return {"tracks": {}, "track_info": {}, "stats": {"success": False}}

        tracks = defaultdict(list)
        track_boxes = {}
        track_info = defaultdict(lambda: {
            "num_frames": 0,
            "first_appearance": None,
            "last_appearance": None,
            "confidences": []
        })
        
        face_counter = 0
        total_faces = 0

        for frame_idx, frame in frames:
            detected = self.detect_faces(frame)
            detected = detected[:max_faces_per_frame]
            
            assignments = self.associate_faces_to_tracks(detected, track_boxes)
            
            for track_id, face_det in assignments.items():
                face = self.extract_face_crop(frame, face_det)
                if face is not None:
                    tracks[track_id].append(face)
                    track_boxes[track_id] = face_det["box"]
                    
                    if track_info[track_id]["first_appearance"] is None:
                        track_info[track_id]["first_appearance"] = frame_idx
                    track_info[track_id]["last_appearance"] = frame_idx
                    track_info[track_id]["num_frames"] += 1
                    track_info[track_id]["confidences"].append(face_det["confidence"])
                    
                    if save_faces:
                        face_filename = f"track_{track_id:02d}_frame_{face_counter:04d}.jpg"
                        face_path = os.path.join(output_dir, face_filename)
                        cv2.imwrite(face_path, face)
                        face_counter += 1
                        total_faces += 1

        for track_id in track_info:
            confidences = track_info[track_id]["confidences"]
            track_info[track_id]["avg_confidence"] = float(np.mean(confidences))
            del track_info[track_id]["confidences"]

        tracks = {tid: faces for tid, faces in tracks.items() if len(faces) > 0}

        return {
            "tracks": dict(tracks),
            "track_info": dict(track_info),
            "stats": {
                "total_frames": len(frames),
                "total_faces": total_faces,
                "num_tracks": len(tracks),
                "success": len(tracks) > 0
            }
        }


if __name__ == "__main__":
    vp = VideoProcessor()
    if vp.load_face_detector():
        print("✅ video_processor multi-face prêt")