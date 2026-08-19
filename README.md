markdown
# Deepfake CPU Benchmark — Fusion Adaptative (sans DANN)

Projet minimal pour mesurer le temps d'inférence réel du modèle **sans DANN** (Fusion adaptative) sur une machine **sans GPU dédié**.  
Ce projet fournit un script en ligne de commande, simple et suffisant pour répondre aux mesures de performance (temps/frame, pipeline exact, runtime, batch size).
---

## 📦 1. Modèles pré-entraînés

Tous les fichiers nécessaires sont disponibles dans le dossier Google Drive partagé :

👉 **[Télécharger l'intégralité des modèles](https://drive.google.com/drive/folders/1GreU_w4CsETa45GV9TK7jOW_VHEkU_Og?usp=sharing)**

Placez l'ensemble de ces fichiers dans le dossier `models/` de ce projet :

| Fichier | Utilité |
|---------|---------|
| `best_val_auc.weights.h5` | Poids du modèle Fusion adaptative entraîné |
| `deploy.prototxt` | Architecture du détecteur de visage SSD |
| `res10_300x300_ssd_iter_140000_fp16.caffemodel` | Poids pré‑entraînés du détecteur SSD |
| `temporal_classifier_fusion_adaptive.pkl` | Classifieur temporel pour l'agrégation des frames |
| `best_threshold_fusion_adaptive.npy` | Seuil optimal pour la classification binaire |

> ⚠️ **Tous ces fichiers sont indispensables** pour obtenir des résultats complets et fiables.

---

## 💻 2. Installation

### Avec Conda (recommandé)

```bash
cd deepfake_cpu_benchmark
conda env create -f environment.yml
conda activate deepfake-cpu-bench
Avec un environnement existant (ex. tf210)
bash
conda activate tf210
pip install -r requirements.txt
🎬 3. Ajouter une vidéo de test
Déposez une ou plusieurs vidéos dans le dossier videos_test/ (formats MP4, AVI, MOV — comme dans l'app originale).
Vous pouvez réutiliser une vidéo de vos jeux de test (par ex. id59_id60_0000.mp4).

🚀 4. Lancer le benchmark
Mode par lot (10 frames par lot – équivalent à l'app d'origine)
bash
python -m src.benchmark --video "videos_test/id60_0006.mp4" --frames 10 --repeats 5
Mode « edge » strict (une frame à la fois, batch_size = 1)
bash
python -m src.benchmark --video "videos_test/This is not Morgan Freeman - A Deepfake Singularity.mp4" --frames 10 --repeats 5 --single-frame
--repeats 5 relance le pipeline 5 fois sur la même vidéo pour lisser les mesures (la première exécution inclut souvent un surcoût d'initialisation TensorFlow/CPU).
Les résultats sont affichés en résumé dans le terminal et sauvegardés dans results/cpu_benchmark_<date>.csv.

📊 5. Ce qui est mesuré – et comment l'interpréter
Le script distingue deux métriques principales :

ms_per_frame_model_forward_only : uniquement le passage avant dans le réseau (les deux branches EfficientNet‑B0 + fusion + classificateur).
C'est le chiffre directement comparable à celui du papier (∼12 ms/frame sur Tesla T4), mesuré ici sur CPU.

ms_per_frame_end_to_end : détection de visage (SSD) + prétraitement + forward pass — soit le temps réel perçu par un utilisateur final.

Les valeurs fps_* sont les inverses respectifs.

🧪 6. Limites connues
Le recadrage du visage se fait par simple crop + marge + redimensionnement, sans alignement par points de repère. Si l'app originale utilisait un alignement par landmarks, ce script en est une approximation (voir src/face_detector.py, docstring de preprocess_for_model). Cela n'a pas d'impact sur la mesure de vitesse, seulement sur le score de confiance affiché pour la vidéo de démonstration.

Une seule machine testée – pas de moyenne inter-machines.

Aucune conversion du modèle (TFLite, ONNX, quantification) : il s'agit de TensorFlow CPU « brut », sans optimisation supplémentaire, afin de fournir une mesure de référence honnête pour l'inférence sur CPU.

📁 7. Structure du projet
text
deepfake_cpu_benchmark/
├── environment.yml           # environnement Conda
├── requirements.txt          # dépendances pip
├── README.md                 # ce fichier
├── models/                   # ⬅️ placez ici les fichiers du Drive
│   ├── best_val_auc.weights.h5
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
│   ├── temporal_classifier_fusion_adaptive.pkl
│   └── best_threshold_fusion_adaptive.npy
├── videos_test/              # vos vidéos de test
├── results/                  # CSV des benchmarks
└── src/
    ├── model_architecture.py   # reconstruction exacte de l'architecture entraînée
    ├── face_detector.py        # détection SSD (réutilisée de l'ancienne app)
    ├── temporal_features.py    # 12 features temporelles (copie exacte du notebook)
    ├── pipeline.py             # pipeline complet + mesure par étape
    └── benchmark.py            # script CLI (point d'entrée)
🧹 Nettoyage de l'environnement Conda (optionnel)
bash
conda env remove -n deepfake-cpu-bench
Bon benchmark !

text

---
