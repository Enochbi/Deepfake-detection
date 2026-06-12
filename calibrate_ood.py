#!/usr/bin/env python3
"""
calibrate_ood.py - Script de calibration OOD rapide

Ce script permet d'ajuster automatiquement le seuil OOD (ood_threshold)
sur la base de données de validation étiquetées.

Prérequis:
- Fichier 'calibration/features_in_domain.npy' : Features de vidéos in-domain (DFDC-val)
- Fichier 'calibration/features_ood.npy' : Features de vidéos OOD (YouTube, animations)

Utilisation:
    python calibrate_ood.py

Résultat:
- Affiche le seuil optimal
- Sauvegarde dans 'models/ood_threshold_calibrated.npy'
- Recommandation conservatrice (entre 0.35 et 0.65)
"""

import numpy as np
import joblib
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


def load_calibration_data():
    """
    Charge les données de calibration
    
    Returns:
        X_in: Features in-domain (N_in, 3) - cosine, euclidean, entropy
        X_ood: Features OOD (N_ood, 3)
    
    Raises:
        FileNotFoundError si fichiers manquants
    """
    calib_dir = "calibration"
    
    in_domain_path = os.path.join(calib_dir, "features_in_domain.npy")
    ood_path = os.path.join(calib_dir, "features_ood.npy")
    
    if not os.path.exists(in_domain_path):
        raise FileNotFoundError(
            f"Fichier in-domain non trouvé: {in_domain_path}\n"
            "Créez ce fichier avec les features OOD calculées sur DFDC-val"
        )
    
    if not os.path.exists(ood_path):
        raise FileNotFoundError(
            f"Fichier OOD non trouvé: {ood_path}\n"
            "Créez ce fichier avec les features OOD calculées sur vidéos hors domaine"
        )
    
    X_in = np.load(in_domain_path)
    X_ood = np.load(ood_path)
    
    print(f"✅ Données chargées:")
    print(f"   In-domain: {X_in.shape[0]} échantillons")
    print(f"   OOD: {X_ood.shape[0]} échantillons")
    
    return X_in, X_ood


def compute_ood_probabilities(X_in, X_ood, ood_detector):
    """
    Calcule les probabilités OOD pour tous les échantillons
    
    Args:
        X_in: Features in-domain (N_in, 3)
        X_ood: Features OOD (N_ood, 3)
        ood_detector: Détecteur OOD chargé (dict avec final_calibrator)
    
    Returns:
        ood_probs: Probabilités OOD pour in+ood (N_in+N_ood,)
        y_true: Labels (0=in-domain, 1=OOD)
    """
    # Concaténer les données
    X_all = np.vstack([X_in, X_ood])
    y_true = np.concatenate([
        np.zeros(len(X_in)),  # 0 = in-domain
        np.ones(len(X_ood))   # 1 = OOD
    ])
    
    # Prédire avec le final_calibrator
    calibrator = ood_detector['final_calibrator']
    ood_probs = calibrator.predict_proba(X_all)[:, 1]
    
    return ood_probs, y_true


def find_optimal_threshold(ood_probs, y_true, method="youden"):
    """
    Trouve le seuil optimal selon différentes métriques
    
    Args:
        ood_probs: Probabilités OOD (N,)
        y_true: Labels vrais (0/1)
        method: "youden" (TPR-FPR max) ou "f1" (F1-score max)
    
    Returns:
        optimal_threshold: Seuil optimal
        metrics: Dict avec métriques à ce seuil
    """
    fpr, tpr, thresholds = roc_curve(y_true, ood_probs)
    roc_auc = auc(fpr, tpr)
    
    if method == "youden":
        # Critère de Youden: maximiser TPR - FPR
        idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[idx]
        optimal_tpr = tpr[idx]
        optimal_fpr = fpr[idx]
        
        print(f"\n📊 Méthode: Youden (TPR - FPR max)")
        
    elif method == "f1":
        # Maximiser F1-score
        precision, recall, pr_thresholds = precision_recall_curve(y_true, ood_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        idx = np.argmax(f1_scores)
        optimal_threshold = pr_thresholds[idx] if idx < len(pr_thresholds) else 0.5
        
        # Retrouver TPR/FPR pour ce seuil
        idx_roc = np.argmin(np.abs(thresholds - optimal_threshold))
        optimal_tpr = tpr[idx_roc]
        optimal_fpr = fpr[idx_roc]
        
        print(f"\n📊 Méthode: F1-score max")
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    metrics = {
        "threshold": optimal_threshold,
        "tpr": optimal_tpr,  # Sensibilité (détection OOD)
        "fpr": optimal_fpr,  # Faux positifs (in-domain classés OOD)
        "specificity": 1 - optimal_fpr,
        "roc_auc": roc_auc
    }
    
    return optimal_threshold, metrics


def plot_calibration_curves(ood_probs, y_true, optimal_threshold, output_path="calibration_curves.png"):
    """
    Génère des graphiques de calibration
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_true, ood_probs)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'r--', lw=1, label='Baseline (random)')
    
    # Marquer le seuil optimal
    idx_opt = np.argmin(np.abs(thresholds - optimal_threshold))
    axes[0].plot(fpr[idx_opt], tpr[idx_opt], 'go', markersize=10, 
                 label=f'Seuil optimal ({optimal_threshold:.3f})')
    
    axes[0].set_xlabel('Taux de faux positifs (FPR)', fontsize=12)
    axes[0].set_ylabel('Taux de vrais positifs (TPR)', fontsize=12)
    axes[0].set_title('Courbe ROC - Détection OOD', fontsize=14)
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # 2. Distribution des scores
    ood_probs_in = ood_probs[y_true == 0]
    ood_probs_ood = ood_probs[y_true == 1]
    
    axes[1].hist(ood_probs_in, bins=30, alpha=0.6, label='In-domain', color='blue', density=True)
    axes[1].hist(ood_probs_ood, bins=30, alpha=0.6, label='OOD', color='red', density=True)
    axes[1].axvline(optimal_threshold, color='green', linestyle='--', lw=2, 
                    label=f'Seuil optimal ({optimal_threshold:.3f})')
    
    axes[1].set_xlabel('Probabilité OOD', fontsize=12)
    axes[1].set_ylabel('Densité', fontsize=12)
    axes[1].set_title('Distribution des scores OOD', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Graphiques sauvegardés: {output_path}")


def apply_conservative_bounds(optimal_threshold, min_threshold=0.35, max_threshold=0.65):
    """
    Applique des bornes conservatrices au seuil
    
    Rationale:
    - min_threshold (0.35): Évite d'être trop permissif (faux négatifs OOD)
    - max_threshold (0.65): Évite d'être trop strict (faux positifs OOD)
    """
    if optimal_threshold < min_threshold:
        print(f"\n⚠️  Seuil optimal ({optimal_threshold:.3f}) < borne min ({min_threshold})")
        print(f"   Recommandation conservatrice: {min_threshold}")
        return min_threshold
    
    if optimal_threshold > max_threshold:
        print(f"\n⚠️  Seuil optimal ({optimal_threshold:.3f}) > borne max ({max_threshold})")
        print(f"   Recommandation conservatrice: {max_threshold}")
        return max_threshold
    
    print(f"\n✅ Seuil optimal dans les bornes conservatrices")
    return optimal_threshold


def calibrate_ood_threshold(method="youden", plot=True):
    """
    Pipeline complet de calibration
    
    Args:
        method: "youden" ou "f1"
        plot: Générer des graphiques
    
    Returns:
        recommended_threshold: Seuil recommandé
    """
    print("=" * 70)
    print("🔧 CALIBRATION DU SEUIL OOD")
    print("=" * 70)
    
    # 1. Charger les données
    try:
        X_in, X_ood = load_calibration_data()
    except FileNotFoundError as e:
        print(f"\n❌ ERREUR: {e}")
        print("\n💡 Conseil:")
        print("   Créez le dossier 'calibration/' avec:")
        print("   - features_in_domain.npy : Features OOD calculées sur DFDC-val")
        print("   - features_ood.npy : Features OOD calculées sur vidéos hors domaine")
        return None
    
    # 2. Charger le détecteur OOD
    ood_detector_path = "models/ensemble_ood_detector_v2_no_maha.pkl"
    if not os.path.exists(ood_detector_path):
        print(f"\n❌ ERREUR: Détecteur OOD non trouvé: {ood_detector_path}")
        return None
    
    print(f"\n📦 Chargement du détecteur OOD...")
    ood_detector = joblib.load(ood_detector_path)
    print("   ✅ Détecteur chargé")
    
    # 3. Calculer les probabilités OOD
    print(f"\n🔮 Calcul des probabilités OOD...")
    ood_probs, y_true = compute_ood_probabilities(X_in, X_ood, ood_detector)
    print(f"   ✅ {len(ood_probs)} prédictions calculées")
    
    # 4. Trouver le seuil optimal
    optimal_threshold, metrics = find_optimal_threshold(ood_probs, y_true, method=method)
    
    print(f"\n   Seuil optimal: {optimal_threshold:.4f}")
    print(f"   TPR (Sensibilité): {metrics['tpr']:.3f} (détection OOD)")
    print(f"   FPR (Faux positifs): {metrics['fpr']:.3f} (in-domain → OOD)")
    print(f"   Spécificité: {metrics['specificity']:.3f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # 5. Appliquer bornes conservatrices
    recommended_threshold = apply_conservative_bounds(optimal_threshold)
    
    # 6. Comparaison avec l'ancien seuil
    old_threshold = 0.25
    print(f"\n📊 COMPARAISON:")
    print(f"   Ancien seuil: {old_threshold}")
    print(f"   Nouveau seuil (brut): {optimal_threshold:.3f}")
    print(f"   ✅ Recommandation finale: {recommended_threshold:.3f}")
    
    if recommended_threshold > old_threshold:
        print(f"\n   ⬆️  Seuil plus strict → Moins de faux négatifs OOD")
    else:
        print(f"\n   ⬇️  Seuil plus permissif → Plus de détections OOD")
    
    # 7. Générer les graphiques
    if plot:
        print(f"\n📊 Génération des graphiques...")
        plot_calibration_curves(ood_probs, y_true, recommended_threshold)
    
    # 8. Sauvegarder le seuil
    output_path = "models/ood_threshold_calibrated.npy"
    np.save(output_path, recommended_threshold)
    print(f"\n💾 Seuil sauvegardé: {output_path}")
    
    # 9. Instructions de mise à jour
    print("\n" + "=" * 70)
    print("📝 INSTRUCTIONS DE MISE À JOUR")
    print("=" * 70)
    print(f"\nDans model_loader.py, ligne ~195:")
    print(f"   AVANT: self.ood_threshold = 0.50")
    print(f"   APRÈS: self.ood_threshold = {recommended_threshold:.4f}")
    print("\nOu bien, chargez dynamiquement:")
    print(f"   threshold_path = 'models/ood_threshold_calibrated.npy'")
    print(f"   self.ood_threshold = np.load(threshold_path)")
    print("=" * 70)
    
    return recommended_threshold


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibration du seuil OOD")
    parser.add_argument(
        "--method", 
        type=str, 
        default="youden", 
        choices=["youden", "f1"],
        help="Méthode de calibration (youden=TPR-FPR max, f1=F1-score max)"
    )
    parser.add_argument(
        "--no-plot", 
        action="store_true",
        help="Ne pas générer de graphiques"
    )
    
    args = parser.parse_args()
    
    recommended_threshold = calibrate_ood_threshold(
        method=args.method,
        plot=not args.no_plot
    )
    
    if recommended_threshold is None:
        print("\n❌ Calibration échouée")
        exit(1)
    
    print(f"\n✅ Calibration réussie: seuil = {recommended_threshold:.4f}")
    exit(0)