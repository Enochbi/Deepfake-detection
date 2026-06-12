"""
model_loader.py - VERSION AVEC RECONSTRUCTION DU MODÈLE
✅ Reconstruction du backbone + fallback load_weights si load_model échoue
✅ Custom layers: fft_layer_gpu, adaptive_fusion, GradientReversal
"""

import os
import gc
import numpy as np
import tensorflow as tf
import joblib
import scipy.stats
from scipy.signal import find_peaks
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics.pairwise import cosine_distances

# ============================================================================
# CONFIG TENSORFLOW
# ============================================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ============================================================================
# FFT LAYER
# ============================================================================
def fft_layer_gpu(x):
    gray = 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
    gray = tf.squeeze(gray, axis=-1)
    gray_complex = tf.cast(gray, tf.complex64)

    fft_result = tf.signal.fft2d(gray_complex)
    fft_shift = tf.signal.fftshift(fft_result)

    magnitude = tf.abs(fft_shift)
    magnitude = tf.maximum(magnitude, 1e-10)

    log_magnitude = tf.math.log(1.0 + magnitude)

    min_val = tf.reduce_min(log_magnitude, axis=[1, 2], keepdims=True)
    max_val = tf.reduce_max(log_magnitude, axis=[1, 2], keepdims=True)

    normalized = tf.where(
        max_val - min_val > 1e-7,
        (log_magnitude - min_val) / (max_val - min_val + 1e-10),
        tf.zeros_like(log_magnitude),
    )

    normalized = tf.clip_by_value(normalized, 0.0, 1.0)
    return tf.stack([normalized, normalized, normalized], axis=-1)

# ============================================================================
# ADAPTIVE FUSION (fonction utilitaire - utilisée comme custom_object si besoin)
# ============================================================================
def adaptive_fusion(inputs):
    w, spatial, freq = inputs
    w = tf.clip_by_value(w, 0.01, 0.99)
    return w * spatial + (1.0 - w) * freq

# ============================================================================
# GRADIENT REVERSAL LAYER (compatible entraînement)
# ============================================================================
@tf.keras.utils.register_keras_serializable()
class GradientReversal(tf.keras.layers.Layer):
    """Gradient Reversal Layer pour DANN (compatible avec la version d'entraînement)"""
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self._lambda = float(lambda_)
        self.lambda_ = None

    def build(self, input_shape):
        # stocke la valeur lambda dans une variable non entraînable pour correspondre à l'entraînement
        self.lambda_ = tf.Variable(
            self._lambda,
            trainable=False,
            dtype=tf.float32,
            name='grl_lambda'
        )
        super().build(input_shape)

    def call(self, x):
        @tf.custom_gradient
        def reverse_grad(x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad
        return reverse_grad(x)

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_": float(self._lambda)})
        return config

# ============================================================================
# RECONSTRUCTION DU MODÈLE (architecture d'évaluation sans branches DANN)
# ============================================================================
def build_hybrid_model_for_eval(input_shape=(224, 224, 3)):
    """
    Reconstruit l'architecture exacte du modèle d'entraînement
    (sans les branches DANN qui ne servent pas en évaluation)
    """
    input_img = layers.Input(shape=input_shape, name="input_image")
    
    # ================= BRANCHE SPATIALE =================
    spatial_backbone = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling="avg",
        name="efficientnetb0_spatial"
    )
    spatial_norm = layers.Rescaling(255.0, name="spatial_rescale")(input_img)
    F_spatial = spatial_backbone(spatial_norm)
    F_spatial = layers.BatchNormalization(name="spatial_bn")(F_spatial)
    F_spatial = layers.Dropout(0.1, name="spatial_dropout")(F_spatial)
    
    # ================= BRANCHE FRÉQUENTIELLE =================
    freq_input = layers.Lambda(fft_layer_gpu, name="fft_transform")(input_img)
    freq_backbone = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling="avg",
        name="efficientnetb0_frequency"
    )
    freq_norm = layers.Rescaling(255.0, name="freq_rescale")(freq_input)
    F_freq = freq_backbone(freq_norm)
    F_freq = layers.BatchNormalization(name="freq_bn")(F_freq)
    F_freq = layers.Dropout(0.1, name="freq_dropout")(F_freq)
    
    # ================= FUSION ADAPTATIVE =================
    concat = layers.Concatenate(name="feature_concat")([F_spatial, F_freq])
    concat = layers.BatchNormalization(name="concat_bn")(concat)
    attention = layers.Dense(
        128, activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="attention_dense"
    )(concat)
    attention = layers.BatchNormalization(name="attention_bn")(attention)
    attention = layers.Dropout(0.2, name="attention_dropout")(attention)
    fusion_weight = layers.Dense(
        1, activation="sigmoid",
        kernel_constraint=tf.keras.constraints.MinMaxNorm(0.0, 1.0),
        name="fusion_weight"
    )(attention)
    
    fused = layers.Lambda(adaptive_fusion, name="adaptive_fusion")(
        [fusion_weight, F_spatial, F_freq]
    )
    fused = layers.BatchNormalization(name="fused_bn")(fused)
    
    # ================= CLASSIFICATEUR =================
    x = layers.Dense(
        256, activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="classifier_dense1"
    )(fused)
    x = layers.BatchNormalization(name="classifier_bn1")(x)
    x = layers.Dropout(0.3, name="classifier_dropout1")(x)
    x = layers.Dense(
        128, activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="classifier_dense2"
    )(x)
    x = layers.BatchNormalization(name="classifier_bn2")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)
    
    model = Model(inputs=input_img, outputs=output, name="hybrid_spatial_frequency")
    return model

# ============================================================================
# FEATURES TEMPORELLES (12)
# ============================================================================
def extract_temporal_features(frame_probs):
    """Extrait 12 features temporelles"""
    probs = np.array(frame_probs)
    n = len(probs)

    if n < 2:
        return np.zeros(12)

    mean = np.mean(probs)
    std = np.std(probs)
    max_p = np.max(probs)
    min_p = np.min(probs)
    range_p = max_p - min_p

    skew = scipy.stats.skew(probs)
    kurt = scipy.stats.kurtosis(probs)

    diffs = np.diff(probs)
    diff_mean = np.mean(diffs)
    diff_std = np.std(diffs)

    peaks, _ = find_peaks(probs, height=0.7)
    drops, _ = find_peaks(1 - probs, height=0.7)

    trend = np.polyfit(np.arange(n), probs, 1)[0]

    features = np.array([
        mean, std, max_p, min_p, range_p,
        skew, kurt,
        diff_mean, diff_std,
        len(peaks) / n,
        len(drops) / n,
        trend
    ])

    return np.nan_to_num(features)

# ============================================================================
# LISSAGE TEMPOREL
# ============================================================================
def temporal_smoothing(values, window=5):
    """Moyenne glissante"""
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    
    return np.array(smoothed)

# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================
class DeepfakeDetector:

    def __init__(
        self,
        frame_weights_path,
        temporal_classifier_path=None,
        threshold_path=None,
        ood_detector_path=None,
        centroid_global_path=None,
        centroid_real_path=None,
        centroid_fake_path=None,
        debug_ood=False
    ):
        self.frame_weights_path = frame_weights_path
        self.temporal_classifier_path = temporal_classifier_path
        self.threshold_path = threshold_path
        
        self.ood_detector_path = ood_detector_path
        self.centroid_global_path = centroid_global_path
        self.centroid_real_path = centroid_real_path
        self.centroid_fake_path = centroid_fake_path

        self.frame_model = None
        self.feature_extractor = None
        self.video_classifier = None
        self.temporal_scaler = None
        self.video_calibrator = None
        self.ood_detector = None
        self.centroid_global = None
        self.centroid_real = None
        self.centroid_fake = None
        
        self.tau_star = 0.3900
        self.ood_threshold = 0.50
        self.reality_threshold_high = 0.60
        self.reality_threshold_low = 0.40
        
        self.image_size = (224, 224)
        self.ood_enabled = False
        self.debug_ood = debug_ood
        
        self.scaler = None
        
        self._load_scalers()

        print("\n🔧 Détecteur initialisé (AGRÉGATION ROBUSTE)")
        print(f"   🔧 tau_star: {self.tau_star}")
        print(f"   🔧 Seuil OOD: {self.ood_threshold}")
        print(f"   🔧 Seuil REAL_HIGH: {self.reality_threshold_high}")
        print(f"   🔧 Seuil REAL_LOW: {self.reality_threshold_low}")

    def _load_scalers(self):
        """Charge les scalers"""
        temporal_scaler_path = "models/temporal_scaler.pkl"
        
        if os.path.exists(temporal_scaler_path):
            try:
                self.temporal_scaler = joblib.load(temporal_scaler_path)
                print("   ✅ Temporal scaler chargé")
            except Exception as e:
                self.temporal_scaler = None

    def load_models(self):
        print("\n🔧 Chargement modèles...")

        custom_objects = {
            "fft_layer_gpu": fft_layer_gpu,
            "adaptive_fusion": adaptive_fusion,
            "GradientReversal": GradientReversal
        }

        # Tentative 1: charger modèle complet (si le .h5 contient l'architecture complète)
        try:
            full_model = tf.keras.models.load_model(
                self.frame_weights_path,
                custom_objects=custom_objects,
                compile=False
            )

            if len(full_model.outputs) > 1:
                self.frame_model = Model(
                    inputs=full_model.input,
                    outputs=full_model.get_layer("output").output
                )
            else:
                self.frame_model = full_model

            print("   ✅ Modèle complet chargé via tf.keras.models.load_model")

        except Exception as e_load:
            print(f"   ⚠️ Échec load_model: {e_load}")
            print("   🔁 Tentative de reconstruction de l'architecture puis load_weights...")

            # Reconstruction de l'architecture
            try:
                model = build_hybrid_model_for_eval(input_shape=(self.image_size[0], self.image_size[1], 3))
                
                # Première tentative: load_weights direct
                try:
                    model.load_weights(self.frame_weights_path)
                    self.frame_model = model
                    print("   ✅ Weights chargés avec model.load_weights (direct).")
                except Exception as e_w1:
                    print(f"   ⚠️ load_weights direct échoué: {e_w1}")
                    # Deuxième tentative: by_name + skip_mismatch
                    try:
                        model.load_weights(self.frame_weights_path, by_name=True, skip_mismatch=True)
                        self.frame_model = model
                        print("   ✅ Weights chargés avec by_name=True, skip_mismatch=True.")
                    except Exception as e_w2:
                        print(f"   ❌ Toutes les tentatives de load_weights ont échoué: {e_w2}")
                        self.frame_model = None

            except Exception as e_recon:
                print(f"   ❌ Échec reconstruction: {e_recon}")
                self.frame_model = None

        # si aucune méthode n'a abouti, lever/retourner False
        if self.frame_model is None:
            print("   ❌ Impossible de charger ou reconstruire le modèle depuis:", self.frame_weights_path)
            return False

        # compile légère pour inference (optionnelle)
        try:
            self.frame_model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="binary_crossentropy"
            )
        except Exception:
            # pas indispensable pour l'inférence si compile échoue
            pass

        # Feature extractor
        try:
            # on tente d'extraire la couche de features fusionnées
            self.feature_extractor = Model(
                inputs=self.frame_model.input,
                outputs=self.frame_model.get_layer("fused_bn").output
            )
        except Exception:
            self.feature_extractor = None

        # Charger classifieur temporel si présent
        if self.temporal_classifier_path and os.path.exists(self.temporal_classifier_path):
            try:
                self.video_classifier = joblib.load(self.temporal_classifier_path)
                print("   ✅ Classifieur temporel chargé")
            except Exception as e:
                self.video_classifier = None
                print(f"   ⚠️ Erreur chargement classifieur temporel: {e}")

        # Charger calibrateur vidéo optionnel
        video_calibrator_path = "models/video_calibrator.pkl"
        if os.path.exists(video_calibrator_path):
            try:
                self.video_calibrator = joblib.load(video_calibrator_path)
                print("   ✅ Video calibrator chargé")
            except Exception as e:
                self.video_calibrator = None
                print(f"   ⚠️ Erreur chargement calibrator: {e}")
        else:
            self.video_calibrator = None
            print("   ℹ️  Pas de video calibrator (utilisation compression conservative)")

        # OOD
        if self.ood_detector_path and os.path.exists(self.ood_detector_path):
            try:
                self.ood_detector = joblib.load(self.ood_detector_path)
                if self.centroid_global_path and os.path.exists(self.centroid_global_path):
                    self.centroid_global = np.load(self.centroid_global_path)
                if self.centroid_real_path and os.path.exists(self.centroid_real_path):
                    self.centroid_real = np.load(self.centroid_real_path)
                if self.centroid_fake_path and os.path.exists(self.centroid_fake_path):
                    self.centroid_fake = np.load(self.centroid_fake_path)
                self.ood_enabled = True
                print("   ✅ OOD chargé")
            except Exception as e:
                self.ood_enabled = False
                print(f"   ⚠️  OOD désactivé: {e}")

        gc.collect()
        return True

    def preprocess_face(self, face_bgr):
        import cv2
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, self.image_size)
        face = face_resized.astype(np.float32) / 255.0
        return np.expand_dims(face, axis=0)

    def compute_ood_features(self, features, probs):
        """Architecture OOD avec normalisation et amplification"""
        if not self.ood_enabled or self.feature_extractor is None:
            return None
        
        n_samples = len(features)
        ood_features = []
        
        for i in range(n_samples):
            feat = features[i].reshape(1, -1)
            prob = probs[i]
            
            if hasattr(self, 'scaler') and self.scaler is not None:
                feat = self.scaler.transform(feat)
            
            cosine_dist = cosine_distances(feat, self.centroid_global.reshape(1, -1))[0, 0]
            euclidean_dist = np.linalg.norm(feat - self.centroid_global.reshape(1, -1))
            
            cosine_dist *= 1.6
            euclidean_dist *= 1.6
            
            p = np.clip(prob, 1e-10, 1 - 1e-10)
            entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
            
            cosine_scaled = self.ood_detector['scaler_cosine'].transform([[cosine_dist]])[0, 0]
            euclidean_scaled = self.ood_detector['scaler_euclidean'].transform([[euclidean_dist]])[0, 0]
            entropy_scaled = self.ood_detector['scaler_entropy'].transform([[entropy]])[0, 0]
            
            ood_features.append([cosine_scaled, euclidean_scaled, entropy_scaled])
        
        return np.array(ood_features)

    def decide_video_final(self, p_video, p_ood):
        """Logique de décision finale"""
        p_real = 1.0 - p_video

        # PRIORITÉ 1: Contenu hors domaine
        if p_ood >= self.ood_threshold:
            return {
                "decision": "OOD_ALERT",
                "confidence": p_ood,
                "reason": f"Hors domaine détecté (p_ood={p_ood:.1%}) - Fiabilité limitée",
                "matrix": (p_video, p_real, p_ood)
            }
        
        # CONTRADICTION
        if p_video >= 0.60 and (1 - p_video) >= 0.60:
            return {
                "decision": "CONTRADICTION_ALERT",
                "confidence": 0.5,
                "reason": "Signaux contradictoires forts",
                "matrix": (p_video, p_real, p_ood)
            }
        
        # FAKE haute confiance
        if p_video >= 0.70:
            return {
                "decision": "FAKE_HIGH_CONFIDENCE",
                "confidence": p_video,
                "reason": f"DEEPFAKE probable (p_fake={p_video:.1%})",
                "matrix": (p_video, p_real, p_ood)
            }
        
        # FAKE selon tau_star
        if p_video >= self.tau_star:
            return {
                "decision": "FAKE",
                "confidence": p_video,
                "reason": f"Suspicion de manipulation (p_fake={p_video:.1%} ≥ τ*={self.tau_star})",
                "matrix": (p_video, p_real, p_ood)
            }
        
        # Vérifier appartenance au domaine
        p_real_domain = 1.0 - p_ood
        
        # REAL haute confiance
        if p_real_domain >= self.reality_threshold_high and p_video <= 0.20:
            return {
                "decision": "REAL_HIGH_CONFIDENCE",
                "confidence": min(p_real_domain, 1 - p_video),
                "reason": f"Appartenance au domaine fort (p_real_domain={p_real_domain:.1%})",
                "matrix": (p_video, p_real, p_ood)
            }
        
        # PROBABLY REAL
        if p_real_domain >= self.reality_threshold_low and p_video <= 0.30:
            return {
                "decision": "PROBABLY_REAL",
                "confidence": p_real_domain,
                "reason": f"Appartenance au domaine modérée (p_real_domain={p_real_domain:.1%})",
                "matrix": (p_video, p_real, p_ood)
            }
        
        # INCERTAIN
        return {
            "decision": "UNKNOWN_CONTENT",
            "confidence": 0.30,
            "reason": f"Contenu ambigu (p_fake={p_video:.1%}, p_real_domain={p_real_domain:.1%})",
            "matrix": (p_video, p_real, p_ood)
        }

    def predict_frames_with_ood(self, face_images):
        """Prédictions frame + OOD"""
        if not face_images:
            return [], [], None
        
        batch = np.vstack([self.preprocess_face(f) for f in face_images])
        frame_probs = self.frame_model.predict(batch, verbose=0).flatten()
        
        ood_probs = np.zeros(len(frame_probs))
        features = None

        try:
            if self.feature_extractor is not None:
                features = self.feature_extractor.predict(batch, verbose=0)
                ood_features = self.compute_ood_features(features, frame_probs)

                if ood_features is not None and self.ood_detector is not None:
                    calibrator = self.ood_detector['final_calibrator']
                    ood_probs = calibrator.predict_proba(ood_features)[:, 1]
        except Exception as e:
            print(f"⚠️ Erreur OOD: {e}")

        return frame_probs.tolist(), ood_probs.tolist(), features

    # ============================================================================
    # ✅ NOUVELLE VERSION: predict_track avec agrégation robuste
    # ============================================================================
    def predict_track(self, track_faces):
        """
        Pipeline complet - AGRÉGATION ROBUSTE & CALIBRATION
        
        Combine:
        1. Classifieur temporel (patterns sur 12 features)
        2. Percentile robuste (capture des pics)
        3. Escalade worst-case (frames extrêmes)
        4. Calibration optionnelle
        """
        if not track_faces:
            return None

        # 1) Frame-level predictions + OOD features
        frame_probs, frame_ood_probs, _ = self.predict_frames_with_ood(track_faces)
        if not frame_probs:
            return None

        # 2) Lissage temporel
        smoothed_probs = temporal_smoothing(frame_probs, window=5)
        smoothed_ood = temporal_smoothing(frame_ood_probs, window=5) if self.ood_enabled else frame_ood_probs

        # ✅ 3) Estimation p_video (DEUX SOURCES):
        #   A) classifieur temporel (si disponible) -> p_clf
        #   B) statistique robuste (percentile 95) -> p_pct
        
        p_clf = None
        temporal_features_dict = {}
        
        if len(smoothed_probs) >= 2 and self.video_classifier is not None:
            temporal_features = extract_temporal_features(smoothed_probs)
            
            # Stocker les features pour l'explication
            temporal_features_dict = {
                "mean": temporal_features[0],
                "std": temporal_features[1],
                "max": temporal_features[2],
                "min": temporal_features[3],
                "range": temporal_features[4],
                "skew": temporal_features[5],
                "kurtosis": temporal_features[6],
                "diff_mean": temporal_features[7],
                "diff_std": temporal_features[8],
                "peaks_norm": temporal_features[9],
                "drops_norm": temporal_features[10],
                "trend": temporal_features[11]
            }
            
            if self.temporal_scaler is not None:
                temporal_features_scaled = self.temporal_scaler.transform(temporal_features.reshape(1, -1))
            else:
                temporal_features_scaled = temporal_features.reshape(1, -1)

            if hasattr(self.video_classifier, "predict_proba"):
                classes = list(self.video_classifier.classes_)
                fake_idx = classes.index(1)   # 1 = FAKE
                p_clf = float(
                    self.video_classifier.predict_proba(temporal_features_scaled)[0, fake_idx]
                )
            else:
                p_clf = float(self.video_classifier.predict(temporal_features_scaled)[0])

        # ✅ Percentile 95: capte les frames très fake
        p_pct = float(np.percentile(smoothed_probs, 95))

        # ✅ 4) Combinaison (conservative blend):
        #    Favoriser le classifieur s'il existe, mais garder la détection de pics
        beta = 0.65 if p_clf is not None else 0.0
        if p_clf is None:
            p_video_raw = p_pct
            combination_method = "percentile_95_only"
        else:
            p_video_raw = float(beta * p_clf + (1.0 - beta) * p_pct)
            combination_method = f"blend(beta={beta:.2f}*clf + {1-beta:.2f}*pct95)"

        # ✅ 5) Escalade si un pic extrême est présent (sécurité):
        #    Si une frame individuelle >= 0.85 fake, on élève p_video_raw au moins jusqu'à 0.85
        max_frame_fake = float(max(smoothed_probs))
        escalation_applied = False
        if max_frame_fake >= 0.85:
            if p_video_raw < 0.85:
                escalation_applied = True
                p_video_raw = max(p_video_raw, 0.85)

        # ✅ 6) Calibration (si un calibrator vidéo existe)
        calibration_method = "none"
        if hasattr(self, "video_calibrator") and self.video_calibrator is not None:
            try:
                p_video = float(self.video_calibrator.predict_proba([[p_video_raw]])[0, 1])
                calibration_method = "trained_calibrator"
            except Exception as e:
                p_video = float(np.clip(p_video_raw, 0.0, 1.0))
                calibration_method = f"fallback_clip (error: {str(e)[:30]})"
        else:
            # Platt-like conservative squashing to avoid overconfidence:
            # apply a mild shrink toward 0.5 if close to extremes
            p_video = float(np.clip(p_video_raw * 0.98 + 0.01, 0.0, 1.0))
            calibration_method = "conservative_compression"

        # 7) OOD summary: percentile 95 sur smoothed_ood
        p_ood_track = float(np.percentile(smoothed_ood, 95)) if len(smoothed_ood) > 0 else 0.0

        # 8) Final decision via decide_video_final
        decision_result = self.decide_video_final(p_video, p_ood_track)

        def _as_list(x):
            if hasattr(x, "tolist"):
                return x.tolist()
            try:
                return list(x)
            except Exception:
                return [x]

        # ✅ Retour avec informations détaillées pour explication
        return {
            "p_video": p_video,
            "p_video_raw": p_video_raw,
            "p_ood": p_ood_track,
            "decision": decision_result["decision"],
            "confidence": decision_result["confidence"],
            "reason": decision_result["reason"],
            "matrix": decision_result["matrix"],
            "num_frames": len(frame_probs),
            "frame_probs": _as_list(smoothed_probs),
            "frame_ood_probs": _as_list(smoothed_ood),
            
            # ✅ Informations détaillées pour explication
            "max_frame_fake": max_frame_fake,
            "p_pct_95": p_pct,
            "p_clf_raw": p_clf,
            "temporal_features": temporal_features_dict,
            "combination_method": combination_method,
            "calibration_method": calibration_method,
            "escalation_applied": escalation_applied
        }

    def predict_video_with_ood(self, face_images):
        """Pipeline vidéo simple"""
        track_result = self.predict_track(face_images)
        
        if track_result is None:
            return None
        
        score_variance = float(np.var(track_result["frame_probs"]))
        
        return {
            "video_prob_fake": track_result["p_video"],
            "video_prediction": track_result["decision"],
            "confidence": track_result["confidence"],
            "reason": track_result["reason"],
            "matrix": track_result["matrix"],
            "ood_prob": track_result["p_ood"],
            "num_frames": track_result["num_frames"],
            "score_variance": score_variance,
            "frame_probs": track_result["frame_probs"],
            "frame_ood_probs": track_result["frame_ood_probs"],
            "max_frame_fake": track_result.get("max_frame_fake", 0.0),
            "p_pct_95": track_result.get("p_pct_95", 0.0),
            "p_clf_raw": track_result.get("p_clf_raw")
        }
