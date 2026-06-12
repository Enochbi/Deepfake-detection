"""
test_ood_corrections.py - VERSION FINALE CORRECTE
Script de validation avec la VRAIE architecture du modèle OOD

Architecture découverte:
- scaler_cosine : Normalise la distance cosine
- scaler_euclidean : Normalise la distance euclidienne
- scaler_entropy : Normalise l'entropie
- final_calibrator : LogisticRegression sur les 3 features normalisées
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_DIR = "models"
OOD_DETECTOR_PATH = f"{MODEL_DIR}/ensemble_ood_detector_v2_no_maha.pkl"
CENTROID_GLOBAL_PATH = f"{MODEL_DIR}/centroid_global.npy"

# ============================================================================
# CALCUL OOD AVEC ARCHITECTURE CORRECTE
# ============================================================================

def compute_ood_score_OLD(feat, centroid_global, prob, ood_model):
    """Version ANCIENNE avec amplifications (incorrecte)"""
    feat = feat.reshape(1, -1) if feat.ndim == 1 else feat
    
    # ❌ ANCIEN: Amplifications arbitraires
    cosine_dist = cosine_distances(feat, centroid_global.reshape(1, -1))[0, 0]
    cosine_feat = cosine_dist * 1.6  # ❌
    
    euclidean_dist = np.linalg.norm(feat - centroid_global.reshape(1, -1))
    euclidean_feat = euclidean_dist * 1.6  # ❌
    
    p = np.clip(prob, 1e-10, 1 - 1e-10)
    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    
    # Normaliser avec les scalers séparés
    cosine_scaled = ood_model['scaler_cosine'].transform([[cosine_feat]])[0, 0]
    euclidean_scaled = ood_model['scaler_euclidean'].transform([[euclidean_feat]])[0, 0]
    entropy_scaled = ood_model['scaler_entropy'].transform([[entropy]])[0, 0]
    
    # Prédiction finale
    features = np.array([[cosine_scaled, euclidean_scaled, entropy_scaled]])
    score = ood_model['final_calibrator'].predict_proba(features)[0, 1]
    
    return score


def compute_ood_score_NEW(feat, centroid_global, prob, ood_model):
    """Version NOUVELLE sans amplifications (correcte)"""
    feat = feat.reshape(1, -1) if feat.ndim == 1 else feat
    
    # ✅ NOUVEAU: Pas d'amplification
    cosine_dist = cosine_distances(feat, centroid_global.reshape(1, -1))[0, 0]
    euclidean_dist = np.linalg.norm(feat - centroid_global.reshape(1, -1))
    
    p = np.clip(prob, 1e-10, 1 - 1e-10)
    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    
    # Normaliser avec les scalers séparés
    cosine_scaled = ood_model['scaler_cosine'].transform([[cosine_dist]])[0, 0]
    euclidean_scaled = ood_model['scaler_euclidean'].transform([[euclidean_dist]])[0, 0]
    entropy_scaled = ood_model['scaler_entropy'].transform([[entropy]])[0, 0]
    
    # Prédiction finale
    features = np.array([[cosine_scaled, euclidean_scaled, entropy_scaled]])
    score = ood_model['final_calibrator'].predict_proba(features)[0, 1]
    
    return score

# ============================================================================
# DIAGNOSTIC
# ============================================================================

def diagnose_ood_model():
    """Analyse le modèle OOD"""
    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC DU MODÈLE OOD")
    print("="*60)
    
    try:
        ood_model = joblib.load(OOD_DETECTOR_PATH)
        centroid_global = np.load(CENTROID_GLOBAL_PATH)
        
        print(f"\n📊 Centroid Global:")
        print(f"   Shape: {centroid_global.shape}")
        print(f"   Mean: {centroid_global.mean():.4f}")
        print(f"   Std: {centroid_global.std():.4f}")
        print("   → Centroid BRUT (pas de normalisation des features avant calcul distances)")
        
        print(f"\n📊 Architecture OOD:")
        print(f"   ✅ scaler_cosine : {ood_model['scaler_cosine'].__class__.__name__}")
        print(f"   ✅ scaler_euclidean : {ood_model['scaler_euclidean'].__class__.__name__}")
        print(f"   ✅ scaler_entropy : {ood_model['scaler_entropy'].__class__.__name__}")
        print(f"   ✅ final_calibrator : {ood_model['final_calibrator'].__class__.__name__}")
        
        return ood_model, centroid_global
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None, None

# ============================================================================
# COMPARAISON
# ============================================================================

def compare_ood_methods(n_samples=1000):
    """Compare les scores OOD avec/sans amplification"""
    print("\n" + "="*60)
    print("🔬 COMPARAISON MÉTHODES OOD")
    print("="*60)
    
    try:
        ood_model, centroid_global = diagnose_ood_model()
        
        if ood_model is None or centroid_global is None:
            return None, None
        
        # Générer features synthétiques
        print(f"\n🧪 Génération de {n_samples} features synthétiques...")
        features = np.random.randn(n_samples, centroid_global.shape[0]) * 0.3 + centroid_global
        probs = np.random.rand(n_samples)
        
        ood_scores_old = []
        ood_scores_new = []
        
        print("⏳ Calcul des scores OOD...")
        for i in range(n_samples):
            feat = features[i]
            prob = probs[i]
            
            # Méthode ANCIENNE
            score_old = compute_ood_score_OLD(feat, centroid_global, prob, ood_model)
            ood_scores_old.append(score_old)
            
            # Méthode NOUVELLE
            score_new = compute_ood_score_NEW(feat, centroid_global, prob, ood_model)
            ood_scores_new.append(score_new)
        
        ood_scores_old = np.array(ood_scores_old)
        ood_scores_new = np.array(ood_scores_new)
        
        # Statistiques
        print(f"\n📊 Résultats sur {n_samples} samples:")
        print(f"\n   ANCIENNE MÉTHODE (avec amplifications * 1.6):")
        print(f"      Min:    {ood_scores_old.min():.4f}")
        print(f"      Max:    {ood_scores_old.max():.4f}")
        print(f"      Mean:   {ood_scores_old.mean():.4f}")
        print(f"      Median: {np.median(ood_scores_old):.4f}")
        print(f"      Std:    {ood_scores_old.std():.4f}")
        
        print(f"\n   NOUVELLE MÉTHODE (sans amplifications):")
        print(f"      Min:    {ood_scores_new.min():.4f}")
        print(f"      Max:    {ood_scores_new.max():.4f}")
        print(f"      Mean:   {ood_scores_new.mean():.4f}")
        print(f"      Median: {np.median(ood_scores_new):.4f}")
        print(f"      Std:    {ood_scores_new.std():.4f}")
        
        diff = ood_scores_old.mean() - ood_scores_new.mean()
        print(f"\n   📉 DIFFÉRENCE:")
        print(f"      Réduction moyenne: {diff:.4f} ({diff/ood_scores_old.mean()*100:.1f}%)")
        
        # Seuils recommandés
        print(f"\n💡 SEUILS RECOMMANDÉS:")
        print(f"\n   ANCIENNE méthode:")
        for p in [90, 92.5, 95, 97.5]:
            print(f"      Percentile {p:4.1f}: {np.percentile(ood_scores_old, p):.4f}")
        
        print(f"\n   NOUVELLE méthode:")
        for p in [90, 92.5, 95, 97.5]:
            print(f"      Percentile {p:4.1f}: {np.percentile(ood_scores_new, p):.4f}")
        
        optimal_old = np.percentile(ood_scores_old, 92.5)
        optimal_new = np.percentile(ood_scores_new, 92.5)
        
        print(f"\n   🎯 SEUIL OPTIMAL (P92.5):")
        print(f"      Ancien: {optimal_old:.4f}")
        print(f"      Nouveau: {optimal_new:.4f} ← À utiliser dans model_loader.py")
        
        # Graphique
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Ancien
            axes[0].hist(ood_scores_old, bins=40, alpha=0.7, color='red', edgecolor='black')
            axes[0].axvline(0.50, color='darkred', linestyle='--', linewidth=2, label='Actuel (0.50)')
            axes[0].axvline(optimal_old, color='green', linestyle='-', linewidth=2, label=f'Optimal ({optimal_old:.2f})')
            axes[0].set_xlabel('OOD Score')
            axes[0].set_ylabel('Fréquence')
            axes[0].set_title('ANCIEN\n(avec amplifications)')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Nouveau
            axes[1].hist(ood_scores_new, bins=40, alpha=0.7, color='green', edgecolor='black')
            axes[1].axvline(0.50, color='darkred', linestyle='--', linewidth=2, label='Actuel (0.50)')
            axes[1].axvline(optimal_new, color='blue', linestyle='-', linewidth=2, label=f'Optimal ({optimal_new:.2f})')
            axes[1].set_xlabel('OOD Score')
            axes[1].set_ylabel('Fréquence')
            axes[1].set_title('NOUVEAU\n(sans amplifications)')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            # Comparaison
            axes[2].boxplot([ood_scores_old, ood_scores_new], labels=['Ancien', 'Nouveau'])
            axes[2].axhline(0.50, color='darkred', linestyle='--', linewidth=2, alpha=0.5)
            axes[2].axhline(optimal_new, color='blue', linestyle='-', linewidth=2, alpha=0.5)
            axes[2].set_ylabel('OOD Score')
            axes[2].set_title('Comparaison')
            axes[2].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ood_comparison.png', dpi=150, bbox_inches='tight')
            print(f"\n📊 Graphique sauvegardé: ood_comparison.png")
            
        except Exception as e:
            print(f"⚠️  Graphique non créé: {e}")
        
        return ood_scores_old, ood_scores_new, optimal_new
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ============================================================================
# RECOMMANDATIONS
# ============================================================================

def generate_recommendations(optimal_threshold):
    """Génère les recommandations finales"""
    print("\n" + "="*60)
    print("✅ CORRECTIONS À APPLIQUER")
    print("="*60)
    
    print(f"\n📝 Dans model_loader.py, fonction compute_ood_features():")
    print(f"\n1️⃣  SUPPRIMER les amplifications:")
    print(f"   ❌ cosine_feat = cosine_dist * 1.6")
    print(f"   ❌ euclidean_feat = euclidean_dist * 1.6")
    print(f"\n   ✅ REMPLACER par:")
    print(f"   cosine_feat = cosine_dist")
    print(f"   euclidean_feat = euclidean_dist")
    
    print(f"\n2️⃣  UTILISER les scalers séparés:")
    print(f"   ✅ REMPLACER tout le code OOD par:")
    print(f"""
    # Calculer les distances (sans amplification)
    cosine_dist = cosine_distances(feat, self.centroid_global.reshape(1, -1))[0, 0]
    euclidean_dist = np.linalg.norm(feat - self.centroid_global.reshape(1, -1))
    
    # Calculer l'entropie
    p = np.clip(prob, 1e-10, 1 - 1e-10)
    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    
    # Normaliser avec les scalers séparés
    cosine_scaled = self.ood_detector['scaler_cosine'].transform([[cosine_dist]])[0, 0]
    euclidean_scaled = self.ood_detector['scaler_euclidean'].transform([[euclidean_dist]])[0, 0]
    entropy_scaled = self.ood_detector['scaler_entropy'].transform([[entropy]])[0, 0]
    
    ood_features.append([cosine_scaled, euclidean_scaled, entropy_scaled])
    """)
    
    print(f"\n3️⃣  DANS predict_frames_with_ood(), utiliser final_calibrator:")
    print(f"   ✅ REMPLACER:")
    print(f"""
    if hasattr(ood_clf, "predict_proba"):
        ood_probs = ood_clf.predict_proba(ood_features)[:, 1]
    """)
    print(f"\n   ✅ PAR:")
    print(f"""
    calibrator = self.ood_detector['final_calibrator']
    ood_probs = calibrator.predict_proba(ood_features)[:, 1]
    """)
    
    print(f"\n4️⃣  AJUSTER le seuil OOD:")
    print(f"   ❌ self.ood_threshold = 0.50")
    print(f"   ✅ self.ood_threshold = {optimal_threshold:.4f}")
    
    print(f"\n5️⃣  GARDER l'agrégation percentile 95:")
    print(f"   ✅ p_ood_track = float(np.percentile(smoothed_ood, 95))")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TEST CORRECTIONS OOD - VERSION FINALE")
    print("="*60)
    
    scores_old, scores_new, optimal = compare_ood_methods(n_samples=1000)
    
    if scores_new is not None:
        # Test sensibilité
        print("\n" + "="*60)
        print("🎯 SENSIBILITÉ AUX SEUILS")
        print("="*60)
        
        print("\n   NOUVELLE MÉTHODE:")
        for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            n_ood = np.sum(scores_new >= threshold)
            pct = (n_ood / len(scores_new)) * 100
            bar = "█" * int(pct / 2)
            marker = " ← OPTIMAL" if abs(threshold - optimal) < 0.02 else ""
            print(f"   Seuil {threshold:.2f}: {n_ood:4d} ({pct:5.1f}%) {bar}{marker}")
        
        generate_recommendations(optimal)
    
    print("\n" + "="*60)
    print("✅ TESTS TERMINÉS")
    print("="*60)
    print("\n💡 PROCHAINES ÉTAPES:")
    print("   1. Copier model_loader_FINAL_V2.py → model_loader.py")
    print("   2. Tester sur vidéos réelles")
    print("   3. Monitorer les performances")
    print()