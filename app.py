"""
app.py - VERSION CORRIGÉE avec résumé explicatif simplifié
✅ Messages simples : Réel / Fake / Incertain
✅ Justification par rapport au seuil
✅ Résumé clair du fonctionnement du modèle
"""

import os
import json
import time
import glob
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

from model_loader import DeepfakeDetector
from video_processor import VideoProcessor

# ============================================================
# Configuration
# ============================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'votre-cle-secrete-ici'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

detector = None
video_processor = None

# ============================================================
# Utilitaires
# ============================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def to_python_type(obj):
    """Convertit NumPy en types Python natifs"""
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj

def cleanup_faces_dir():
    """Nettoie le dossier des visages"""
    faces_dir = "static/faces"
    if not os.path.exists(faces_dir):
        return
    deleted = 0
    for f in os.listdir(faces_dir):
        path = os.path.join(faces_dir, f)
        if os.path.isfile(path):
            try:
                os.remove(path)
                deleted += 1
            except:
                pass
    if deleted > 0:
        print(f"🧹 {deleted} image(s) de visages nettoyée(s)")

def cleanup_uploads_dir():
    """Nettoie les vidéos uploadées"""
    uploads_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(uploads_dir):
        return
    
    current_time = time.time()
    deleted = 0
    
    for f in os.listdir(uploads_dir):
        path = os.path.join(uploads_dir, f)
        if os.path.isfile(path):
            if current_time - os.path.getmtime(path) > 3600:
                try:
                    os.remove(path)
                    deleted += 1
                except:
                    pass
    
    if deleted > 0:
        print(f"🧹 {deleted} vidéo(s) obsolète(s) nettoyée(s)")

def cleanup_all():
    """Nettoyage complet"""
    cleanup_faces_dir()
    cleanup_uploads_dir()

# ============================================================
# ✅ FONCTION SIMPLIFIÉE POUR RÉSUMÉ TECHNIQUE
# ============================================================
def calculate_technical_summary(video_decision):
    """
    Version simplifiée qui retourne uniquement les données nécessaires pour l'explication
    
    Args:
        video_decision: Dictionnaire contenant tous les résultats de l'analyse vidéo
        
    Returns:
        Dictionnaire avec les données pour l'affichage
    """
    
    # Extraire les données depuis video_decision
    p_video = video_decision.get('p_video', 0.0)
    p_video_raw = video_decision.get('p_video_raw', p_video)
    temporal_features = video_decision.get('temporal_features', {})
    p_pct_95 = video_decision.get('p_pct_95', 0.0)
    max_frame_fake = video_decision.get('max_frame_fake', 0.0)
    
    # Extraire les features temporelles
    std = temporal_features.get('std', 0.1)
    mean = temporal_features.get('mean', 0)
    max_val = temporal_features.get('max', 0)
    min_val = temporal_features.get('min', 0)
    range_val = temporal_features.get('range', 0)
    skew = temporal_features.get('skew', 0)
    kurtosis = temporal_features.get('kurtosis', 0)
    diff_std = temporal_features.get('diff_std', 0)
    tau_star = 0.39
    
    return {
        # Scores principaux
        'score_brut': round(p_video_raw, 4),
        'score_brut_pct': round(p_video_raw * 100, 1),
        'p_final': round(p_video, 4),
        'p_final_pct': round(p_video * 100, 1),
        
        # Features temporelles pour l'affichage
        'std': round(std, 4),
        'mean': round(mean, 4),
        'max': round(max_val, 4),
        'min': round(min_val, 4),
        'range': round(range_val, 4),
        'skew': round(skew, 4),
        'kurtosis': round(kurtosis, 4),
        'diff_std': round(diff_std, 4),
        
        # Paramètres du modèle
        'tau_star': tau_star,
        'p_pct_95': round(p_pct_95, 4),
        'max_frame_fake': round(max_frame_fake, 4)
    }

# ============================================================
# ✅ MESSAGES SIMPLIFIÉS
# ============================================================
def generate_simple_decision(p_video, tau_star):
    """
    Décision simplifiée : Réel / Fake / Incertain
    """
    p_video_pct = round(p_video * 100, 1)
    tau_pct = round(tau_star * 100, 1)
    
    if p_video >= 0.70:
        return {
            'decision': 'Fake',
            'message': f'Suspicion de deepfake détectée',
            'justification': f'Score de {p_video_pct}% dépasse largement le seuil de {tau_pct}%',
            'color': 'fake'
        }
    elif p_video >= tau_star:
        return {
            'decision': 'Fake',
            'message': f'Probabilité de manipulation',
            'justification': f'Score de {p_video_pct}% supérieur au seuil de {tau_pct}%',
            'color': 'fake'
        }
    elif p_video <= 0.20:
        return {
            'decision': 'Réel',
            'message': f'Vidéo probablement authentique',
            'justification': f'Score de {p_video_pct}% bien inférieur au seuil de {tau_pct}%',
            'color': 'real'
        }
    elif p_video <= 0.30:
        return {
            'decision': 'Réel',
            'message': f'Faible suspicion de manipulation',
            'justification': f'Score de {p_video_pct}% reste sous le seuil de {tau_pct}%',
            'color': 'real'
        }
    else:
        return {
            'decision': 'Décision Incertaine',
            'message': f'Résultat ambigu - Vérification recommandée',
            'justification': f'Score de {p_video_pct}% proche du seuil de {tau_pct}%',
            'color': 'uncertain'
        }

# ============================================================
# ✅ AGRÉGATION MULTI-TRACK
# ============================================================
def aggregate_video_decision(track_results, all_faces):
    """Agrégation robuste pour multi-tracks"""
    if not track_results:
        return None

    num_tracks = len(track_results)

    if num_tracks == 1:
        track = track_results[0]
        return {
            "p_video": track["p_video"],
            "p_video_raw": track.get("p_video_raw", track["p_video"]),
            "p_ood": track["p_ood"],
            "num_tracks": 1,
            "tracks": track_results,
            "aggregation_method": "single_track",
            "temporal_features": track.get("temporal_features", {}),
            "p_clf_raw": track.get("p_clf_raw"),
            "p_pct_95": track.get("p_pct_95"),
            "max_frame_fake": track.get("max_frame_fake"),
            "combination_method": track.get("combination_method"),
            "calibration_method": track.get("calibration_method"),
            "escalation_applied": track.get("escalation_applied", False),
            "frame_probs": track.get("frame_probs", [])
        }

    print(f"\n🔄 Multi-tracks détecté: {num_tracks} visages")

    total_frames = sum(t["num_frames"] for t in track_results)
    if total_frames > 0:
        weighted_avg = sum(t["p_video"] * t["num_frames"] for t in track_results) / total_frames
    else:
        weighted_avg = float(np.mean([t["p_video"] for t in track_results]))

    worst = max(track_results, key=lambda x: x["p_video"])
    worst_p = worst["p_video"]

    global_result = detector.predict_track(all_faces)
    global_p = global_result["p_video"] if global_result is not None else None

    candidates = [weighted_avg, worst_p]
    if global_p is not None:
        candidates.append(global_p)

    p_video_final = float(max(candidates))

    p_ood_final = max(
        [t.get("p_ood", 0.0) for t in track_results] +
        ([global_result.get("p_ood", 0.0)] if global_result else [])
    )

    source_info = global_result if global_result is not None else worst

    all_frame_probs = []
    for t in track_results:
        all_frame_probs.extend(t.get('frame_probs', []))

    return {
        "p_video": p_video_final,
        "p_video_raw": source_info.get("p_video_raw", p_video_final),
        "p_ood": p_ood_final,
        "num_tracks": num_tracks,
        "tracks": track_results,
        "aggregation_method": f"robust_max",
        "temporal_features": source_info.get("temporal_features", {}),
        "p_clf_raw": source_info.get("p_clf_raw"),
        "p_pct_95": source_info.get("p_pct_95"),
        "max_frame_fake": source_info.get("max_frame_fake"),
        "combination_method": source_info.get("combination_method"),
        "calibration_method": source_info.get("calibration_method"),
        "escalation_applied": source_info.get("escalation_applied", False),
        "frame_probs": all_frame_probs
    }

# ============================================================
# Initialisation
# ============================================================
def init_models():
    global detector, video_processor

    print("\n" + "=" * 60)
    print("🚀 INITIALISATION SYSTÈME - VERSION SIMPLIFIÉE")
    print("=" * 60)
    
    print("\n🧹 Nettoyage des fichiers temporaires...")
    cleanup_all()

    print("\n1️⃣  Chargement du détecteur de visages...")
    video_processor = VideoProcessor(model_dir="models", min_confidence=0.5, iou_threshold=0.3)
    if not video_processor.load_face_detector():
        print("❌ Échec du chargement du détecteur de visages")
        return False

    print("\n2️⃣  Chargement des modèles...")
    
    frame_weights = "models/best_val_auc_20260103_201316.h5"
    temporal_clf = "models/temporal_classifier.pkl"
    threshold_file = "models/best_threshold.npy"
    
    ood_detector_file = "models/ensemble_ood_detector_v2_no_maha.pkl"
    centroid_global_file = "models/centroid_global.npy"
    centroid_real_file = "models/centroid_real.npy"
    centroid_fake_file = "models/centroid_fake.npy"
    
    if not os.path.exists(frame_weights):
        candidates = glob.glob("models/*.h5")
        frame_weights = candidates[0] if candidates else None
        if not frame_weights:
            print("❌ Aucun fichier de poids frame-level trouvé")
            return False
    
    if not os.path.exists(temporal_clf):
        print("⚠️  Classifieur temporel non trouvé")
        temporal_clf = None
    
    if not os.path.exists(threshold_file):
        print("⚠️  Seuil optimal non trouvé, utilisation de tau_star=0.39")
        threshold_file = None
    
    ood_available = all([
        os.path.exists(ood_detector_file),
        os.path.exists(centroid_global_file),
        os.path.exists(centroid_real_file),
        os.path.exists(centroid_fake_file)
    ])
    
    if not ood_available:
        print("⚠️  Fichiers OOD manquants - Détection OOD désactivée")
        ood_detector_file = None
    
    detector = DeepfakeDetector(
        frame_weights_path=frame_weights,
        temporal_classifier_path=temporal_clf,
        threshold_path=threshold_file,
        ood_detector_path=ood_detector_file if ood_available else None,
        centroid_global_path=centroid_global_file if ood_available else None,
        centroid_real_path=centroid_real_file if ood_available else None,
        centroid_fake_path=centroid_fake_file if ood_available else None
    )
    
    if not detector.load_models():
        print("❌ Échec du chargement des modèles")
        return False

    print("\n✅ SYSTÈME PRÊT")
    return True

# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/js', filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'Aucun fichier vidéo fourni'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format non autorisé'}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        return jsonify({
            'success': True,
            'message': 'Vidéo uploadée avec succès',
            'filename': unique_filename
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Requête JSON invalide'}), 400

        filename = data.get('filename')
        max_frames = min(int(data.get('max_frames', 10)), 30)

        if not filename:
            return jsonify({'success': False, 'error': 'Nom de fichier manquant'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Fichier non trouvé'}), 404

        print("\n🧹 Nettoyage des anciennes données...")
        cleanup_all()
        
        start_time = time.time()

        try:
            print(f"\n📹 Traitement: {filename}")
            result = video_processor.process_video(
                filepath,
                max_frames=max_frames,
                max_faces_per_frame=5,
                save_faces=True,
                output_dir='static/faces'
            )

            if not result['stats'].get('success') or not result['tracks']:
                return jsonify({
                    'success': False,
                    'error': 'Aucun visage détecté dans la vidéo',
                    'details': result['stats']
                }), 400

            num_tracks = result['stats']['num_tracks']
            print(f"✅ {num_tracks} track(s) détecté(s)")

            print("🔍 Analyse individuelle par track...")
            track_results = []
            all_faces_combined = []
            
            for track_id, track_faces in result['tracks'].items():
                print(f"   Track #{track_id}: {len(track_faces)} frames")
                track_result = detector.predict_track(track_faces)
                
                if track_result:
                    track_result['track_id'] = track_id
                    track_results.append(track_result)
                    print(f"   ✓ Track #{track_id}: p_video={track_result['p_video']:.3f}")
                
                all_faces_combined.extend(track_faces)

            print("\n🎯 Agrégation finale...")
            video_decision = aggregate_video_decision(track_results, all_faces_combined)
            
            if video_decision is None:
                return jsonify({
                    'success': False,
                    'error': 'Erreur lors de l\'analyse'
                }), 500
            
            p_video = video_decision['p_video']
            p_video_raw = video_decision.get('p_video_raw', p_video)
            
            print(f"✅ Score final: {p_video:.3f}")

            # ✅ CALCUL DU RÉSUMÉ TECHNIQUE SIMPLIFIÉ
            technical_formula = calculate_technical_summary(video_decision)
            
            # Décision simple
            simple_decision = generate_simple_decision(p_video, detector.tau_star)

            processing_time = time.time() - start_time

            # Préparer les visages pour l'UI
            faces_dir = 'static/faces'
            face_images_urls = []
            face_probs = []
            
            if os.path.exists(faces_dir):
                for track_id in sorted(result['tracks'].keys()):
                    track_result = next((t for t in track_results if t['track_id'] == track_id), None)
                    if track_result:
                        track_files = sorted([
                            f for f in os.listdir(faces_dir) 
                            if f.startswith(f"track_{track_id:02d}_")
                        ])
                        
                        for i, face_file in enumerate(track_files[:3]):
                            face_images_urls.append(f"/static/faces/{face_file}")
                            if i < len(track_result.get('frame_probs', [])):
                                face_probs.append(track_result['frame_probs'][i])

            response = {
                'success': True,
                
                # Décision
                'simple_decision': simple_decision,
                
                # ✅ RÉSUMÉ TECHNIQUE SIMPLIFIÉ
                'technical_formula': technical_formula,
                
                # Scores
                'prob_fake': round(p_video, 4),
                'prob_real': round(1.0 - p_video, 4),
                'prob_fake_percent': round(min(p_video * 100, 99.9), 1),
                'prob_real_percent': round(min((1.0 - p_video) * 100, 99.9), 1),
                'score_raw_percent': round(p_video_raw * 100, 1),
                
                # Seuils
                'threshold': detector.tau_star,
                'threshold_percent': round(detector.tau_star * 100, 1),
                
                # Stats
                'processing_time': round(processing_time, 2),
                'num_frames': sum(t['num_frames'] for t in track_results),
                'num_tracks': video_decision['num_tracks'],
                'num_faces': result['stats']['total_faces'],
                
                # Frame-level
                'frame_probs_fake': video_decision.get('frame_probs', []),
                
                # Visages
                'faces_data': [
                    {
                        'image_url': url,
                        'probability': prob
                    }
                    for url, prob in zip(face_images_urls, face_probs)
                ],
                
                # Détails
                'calculation_details': {
                    'p_video_raw': round(p_video_raw, 4),
                    'p_video_calibrated': round(p_video, 4),
                    'p_pct_95': video_decision.get('p_pct_95'),
                    'max_frame_fake': video_decision.get('max_frame_fake'),
                    'aggregation_method': video_decision.get('aggregation_method')
                }
            }

            response = to_python_type(response)

            # Sauvegarder
            result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}.json"
            with open(os.path.join('results', result_filename), 'w') as f:
                json.dump(response, f, indent=2)
            
            return jsonify(response), 200
            
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"🧹 Vidéo supprimée: {filename}")
                except Exception as e:
                    print(f"⚠️  Erreur suppression: {e}")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Erreur interne lors de l\'analyse',
            'details': str(e)
        }), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'version': 'simplified',
        'detector_loaded': detector is not None,
        'video_processor_loaded': video_processor is not None,
        'tau_star': detector.tau_star if detector else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory('results', filename)

@app.route("/cleanup", methods=["POST"])
def cleanup():
    try:
        cleanup_all()
        return jsonify({"status": "ok", "message": "Nettoyage effectué"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================================
# Point d'entrée
# ============================================================
if __name__ == '__main__':
    if not init_models():
        print("❌ Impossible d'initialiser les modèles. Arrêt.")
        exit(1)

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)