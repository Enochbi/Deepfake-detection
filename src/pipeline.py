"""
Pipeline complet de detection, avec mesure de temps par etape.

Reproduit le pipeline decrit au Chapitre 3.3.1 du memoire :
video -> echantillonnage de N frames -> detection de visage (SSD) ->
crop+resize 224x224 -> inference frame-level (modele hybride) ->
agregation temporelle (12 features + LogisticRegression) -> verdict.

Le classifieur temporel est OPTIONNEL : s'il n'est pas fourni (fichier
.pkl absent), le pipeline bascule sur un repli explicite (moyenne des
probabilites frame-level) et l'indique clairement dans les resultats
(is_official_temporal_classifier=False), pour ne jamais confondre ce
repli avec le vrai resultat du papier.
"""
import os
import time
import joblib
import numpy as np
import cv2

from .temporal_features import extract_temporal_features


class DeepfakeCPUPipeline:
    def __init__(self, model, face_detector, temporal_classifier_path=None,
                 threshold_path=None, fallback_threshold=0.5):
        self.model = model
        self.face_detector = face_detector

        self.temporal_clf = None
        self.has_official_temporal_clf = False
        if temporal_classifier_path and os.path.exists(temporal_classifier_path):
            self.temporal_clf = joblib.load(temporal_classifier_path)
            self.has_official_temporal_clf = True

        self.threshold = fallback_threshold
        self.has_official_threshold = False
        if threshold_path and os.path.exists(threshold_path):
            self.threshold = float(np.load(threshold_path))
            self.has_official_threshold = True

        if not self.has_official_temporal_clf:
            print("⚠️  Classifieur temporel officiel introuvable "
                  "(temporal_classifier_fusion_adaptive.pkl manquant dans models/). "
                  "Repli sur p_video = moyenne des p_frame -- NE PAS CITER comme "
                  "resultat du papier, ceci est un mode degrade pour permettre le "
                  "benchmark de vitesse en son absence.")
        if not self.has_official_threshold:
            print(f"⚠️  Seuil officiel introuvable (best_threshold_fusion_adaptive.npy "
                  f"manquant). Seuil de repli utilise: {self.threshold}")

    @staticmethod
    def sample_frames(video_path, n_frames=10):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"Impossible de lire {video_path} (0 frame detectee)")
        n = min(n_frames, total)
        indices = np.linspace(0, total - 1, n).astype(int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()
        return frames

    def run(self, video_path, n_frames=10, batch_inference=True):
        """Execute le pipeline complet sur une video et retourne un dict de
        resultats + un dict de timings detailles (en secondes)."""
        timings = {"face_detection": [], "preprocessing": [], "model_forward": []}

        t0 = time.perf_counter()
        frames = self.sample_frames(video_path, n_frames=n_frames)
        t_extract = time.perf_counter() - t0

        faces, confidences = [], []
        for frame in frames:
            t1 = time.perf_counter()
            crop, conf = self.face_detector.detect_best_face(frame)
            timings["face_detection"].append(time.perf_counter() - t1)
            if crop is None:
                continue

            t2 = time.perf_counter()
            face_arr = self.face_detector.preprocess_for_model(crop)
            timings["preprocessing"].append(time.perf_counter() - t2)

            faces.append(face_arr)
            confidences.append(conf)

        if not faces:
            raise RuntimeError("Aucun visage detecte dans les frames echantillonnees.")

        faces_batch = np.stack(faces, axis=0)  # (N, 224, 224, 3)

        if batch_inference:
            t3 = time.perf_counter()
            frame_probs = self.model.predict(faces_batch, verbose=0).ravel()
            timings["model_forward"].append(time.perf_counter() - t3)
        else:
            frame_probs = []
            for f in faces_batch:
                t3 = time.perf_counter()
                p = self.model.predict(f[np.newaxis, ...], verbose=0)[0, 0]
                timings["model_forward"].append(time.perf_counter() - t3)
                frame_probs.append(p)
            frame_probs = np.array(frame_probs)

        if self.has_official_temporal_clf:
            feats = extract_temporal_features(frame_probs).reshape(1, -1)
            p_video = float(self.temporal_clf.predict_proba(feats)[0, 1])
        else:
            p_video = float(np.mean(frame_probs))

        verdict = "FAKE" if p_video >= self.threshold else "REAL"

        result = {
            "video_path": video_path,
            "n_frames_requested": n_frames,
            "n_frames_with_face": len(faces),
            "frame_probs": frame_probs.tolist(),
            "p_video": p_video,
            "threshold": self.threshold,
            "verdict": verdict,
            "used_official_temporal_classifier": self.has_official_temporal_clf,
            "used_official_threshold": self.has_official_threshold,
        }
        timings["frame_extraction_total"] = t_extract
        return result, timings
