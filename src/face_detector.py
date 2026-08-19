"""
Detecteur de visages SSD (res10_300x300_ssd_iter_140000_fp16), reutilise
depuis l'application de demonstration precedente (cf. rapport, Chapitre 3.3.1).

Meme detecteur que l'app originale : rapide, leger, adapte au CPU.
"""
import cv2
import numpy as np

FACE_TARGET_SIZE = (224, 224)
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MARGIN = 0.20  # marge ajoutee autour de la boite detectee avant crop


class SSDFaceDetector:
    def __init__(self, prototxt_path, caffemodel_path,
                 confidence_threshold=DEFAULT_CONFIDENCE, margin=DEFAULT_MARGIN):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        self.confidence_threshold = confidence_threshold
        self.margin = margin

    def detect_best_face(self, frame_bgr):
        """Retourne le crop du visage le plus confiant (BGR, non redimensionne),
        ou None si aucun visage n'est detecte au-dessus du seuil."""
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        best_conf, best_box = 0.0, None
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold or confidence <= best_conf:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_conf, best_box = confidence, box.astype(int)

        if best_box is None:
            return None, 0.0

        x1, y1, x2, y2 = best_box
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * self.margin), int(bh * self.margin)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        crop = frame_bgr[y1:y2, x1:x2]
        return crop, best_conf

    @staticmethod
    def preprocess_for_model(face_crop_bgr, target_size=FACE_TARGET_SIZE):
        """BGR crop -> RGB [0,1] float32, redimensionne, pret pour le modele.

        NOTE (a verifier / documenter dans le rapport) : ceci est un simple
        resize apres crop+marge, SANS alignement par points de repere
        (landmarks). Le chapitre 3 du memoire mentionne des visages
        "alignes" sans preciser la methode ; si l'alignement original
        utilisait une rotation base sur les yeux, ce module est une
        approximation plus simple. Cela peut legerement affecter les scores
        de confiance absolus, mais n'affecte pas la mesure de temps
        d'inference (objet de ce benchmark).
        """
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
        return (resized.astype(np.float32) / 255.0)
