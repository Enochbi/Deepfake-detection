"""
Benchmark CPU du pipeline complet (sans DANN), pour repondre au commentaire
du Reviewer 2 sur le deploiement "edge" / sans materiel specialise.

Usage:
    python -m src.benchmark --video videos_test/exemple.mp4
    python -m src.benchmark --video videos_test/exemple.mp4 --frames 10 --repeats 5
    python -m src.benchmark --video videos_test/exemple.mp4 --single-frame

Repond explicitement, a la fin, aux 4 questions :
    1) Temps d'inference mesure (ms/frame, FPS)
    2) Quel pipeline (frame-level seul, ou + agregation temporelle)
    3) Quel runtime (TensorFlow CPU, pas de conversion TFLite/ONNX ici)
    4) Batch size utilise (1 = single-frame, ou lot de N)
"""
import os
# Force l'utilisation du CPU meme si un GPU/accelerateur etait detecte,
# pour garantir que la mesure correspond bien a un scenario "sans GPU dedie".
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import csv
import platform
import statistics
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from .model_architecture import load_trained_model
from .face_detector import SSDFaceDetector
from .pipeline import DeepfakeCPUPipeline

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def build_pipeline():
    weights_path = os.path.join(MODELS_DIR, "best_val_auc.weights.h5")
    prototxt_path = os.path.join(MODELS_DIR, "deploy.prototxt")
    caffemodel_path = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    temporal_clf_path = os.path.join(MODELS_DIR, "temporal_classifier_fusion_adaptive.pkl")
    threshold_path = os.path.join(MODELS_DIR, "best_threshold_fusion_adaptive.npy")

    for required in [weights_path, prototxt_path, caffemodel_path]:
        if not os.path.exists(required):
            raise FileNotFoundError(
                f"Fichier requis manquant: {required}\n"
                f"Voir README.md, section 'Fichiers a placer dans models/'."
            )

    print("⏳ Construction de l'architecture + chargement des poids...")
    t0 = time.perf_counter()
    model = load_trained_model(weights_path)
    load_time = time.perf_counter() - t0
    print(f"✅ Modele charge en {load_time:.2f}s "
          f"({model.count_params():,} parametres)")

    face_detector = SSDFaceDetector(prototxt_path, caffemodel_path)

    pipeline = DeepfakeCPUPipeline(
        model=model,
        face_detector=face_detector,
        temporal_classifier_path=temporal_clf_path,
        threshold_path=threshold_path,
    )
    return pipeline, load_time


def run_benchmark(video_path, n_frames, repeats, single_frame):
    pipeline, model_load_time = build_pipeline()
    batch_inference = not single_frame

    print(f"\n▶️  Benchmark: {repeats} repetitions, {n_frames} frames/video, "
          f"mode={'single-frame (batch=1)' if single_frame else f'batch={n_frames}'}\n")

    all_face_det, all_preproc, all_forward, all_total = [], [], [], []
    result = None
    for run_idx in range(repeats):
        t_start = time.perf_counter()
        result, timings = pipeline.run(video_path, n_frames=n_frames,
                                        batch_inference=batch_inference)
        t_total = time.perf_counter() - t_start

        all_face_det.extend(timings["face_detection"])
        all_preproc.extend(timings["preprocessing"])
        all_forward.extend(timings["model_forward"])
        all_total.append(t_total)
        print(f"   run {run_idx + 1}/{repeats}: {t_total*1000:.1f} ms total "
              f"({result['n_frames_with_face']} visages traites)")

    n_frames_total = result["n_frames_with_face"] * repeats
    forward_total_s = sum(all_forward)
    ms_per_frame_forward = (forward_total_s / n_frames_total) * 1000
    ms_per_frame_face_det = (sum(all_face_det) / n_frames_total) * 1000
    ms_per_frame_preproc = (sum(all_preproc) / n_frames_total) * 1000
    ms_per_frame_end_to_end = ms_per_frame_forward + ms_per_frame_face_det + ms_per_frame_preproc
    fps_forward_only = 1000.0 / ms_per_frame_forward
    fps_end_to_end = 1000.0 / ms_per_frame_end_to_end

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "tensorflow_version": tf.__version__,
        "runtime": "TensorFlow CPU (aucune conversion TFLite/ONNX)",
        "pipeline": ("frame-level + agregation temporelle (LogisticRegression officiel)"
                     if result["used_official_temporal_classifier"]
                     else "frame-level + repli moyenne (classifieur temporel officiel absent)"),
        "batch_mode": "single-frame (batch_size=1)" if single_frame else f"batch (batch_size={n_frames})",
        "n_frames_per_video": n_frames,
        "n_repeats": repeats,
        "model_load_time_s": round(model_load_time, 3),
        "ms_per_frame_model_forward_only": round(ms_per_frame_forward, 2),
        "ms_per_frame_face_detection": round(ms_per_frame_face_det, 2),
        "ms_per_frame_preprocessing": round(ms_per_frame_preproc, 2),
        "ms_per_frame_end_to_end": round(ms_per_frame_end_to_end, 2),
        "fps_model_forward_only": round(fps_forward_only, 1),
        "fps_end_to_end": round(fps_end_to_end, 1),
        "total_video_time_mean_s": round(statistics.mean(all_total), 3),
        "total_video_time_std_s": round(statistics.stdev(all_total), 3) if len(all_total) > 1 else 0.0,
        "last_verdict": result["verdict"],
        "last_p_video": round(result["p_video"], 4),
    }
    return summary


def print_answers(summary):
    print("\n" + "=" * 78)
    print("REPONSES AUX 4 QUESTIONS (a reporter dans le rapport / mail encadrant)")
    print("=" * 78)
    print(f"1) Temps d'inference mesure :")
    print(f"   - Forward pass modele seul : {summary['ms_per_frame_model_forward_only']} ms/frame "
          f"({summary['fps_model_forward_only']} FPS)")
    print(f"   - Pipeline complet (detection visage + pretraitement + modele) : "
          f"{summary['ms_per_frame_end_to_end']} ms/frame ({summary['fps_end_to_end']} FPS)")
    print(f"2) Pipeline mesure : {summary['pipeline']}")
    print(f"3) Runtime : {summary['runtime']} (TensorFlow {summary['tensorflow_version']})")
    print(f"4) Mode : {summary['batch_mode']}")
    print("=" * 78)


def save_csv(summary, out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"cpu_benchmark_{ts}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])
    print(f"\n💾 Resultats sauvegardes: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU du pipeline de detection (sans DANN)")
    parser.add_argument("--video", required=True, help="Chemin vers une video de test")
    parser.add_argument("--frames", type=int, default=10,
                         help="Nombre de frames echantillonnees par video (defaut: 10, comme l'app originale)")
    parser.add_argument("--repeats", type=int, default=5,
                         help="Nombre de repetitions pour stabiliser la mesure (defaut: 5)")
    parser.add_argument("--single-frame", action="store_true",
                         help="Force l'inference une frame a la fois (batch_size=1) au lieu d'un lot")
    args = parser.parse_args()

    summary = run_benchmark(args.video, args.frames, args.repeats, args.single_frame)
    print_answers(summary)
    save_csv(summary)


if __name__ == "__main__":
    main()
