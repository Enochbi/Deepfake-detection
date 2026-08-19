# Deepfake CPU Benchmark — Variante Fusion adaptative (sans DANN)

Projet minimal pour mesurer le temps d'inférence réel du modèle **sans DANN**
(Fusion adaptative) sur une machine **sans GPU dédié**, en réponse au
commentaire du Reviewer 2 sur les affirmations de déploiement "edge" /
sans matériel spécialisé.

Ce projet **ne réutilise pas l'app Flask complète** — il fournit un script
en ligne de commande, plus simple et suffisant pour répondre aux 4
questions de mesure (temps/frame, pipeline exact, runtime, batch size).
Vous pourrez brancher ces mêmes modules (`src/model_architecture.py`,
`src/face_detector.py`, `src/pipeline.py`) dans l'app Flask existante
si vous voulez aussi la redéployer telle quelle.

## 1. Fichiers à placer dans `models/`

À copier depuis vos exports Kaggle / anciens fichiers d'app :

| Fichier | Source | Obligatoire ? |
|---|---|---|
| `best_val_auc.weights.h5` | Zip `FUSION_ADAPTIVE_DIAGNOSTIC_RESULTS/models/` | ✅ Oui |
| `deploy.prototxt` | Ancienne app (déjà en votre possession) | ✅ Oui |
| `res10_300x300_ssd_iter_140000_fp16.caffemodel` | Ancienne app (déjà en votre possession) | ✅ Oui |
| `temporal_classifier_fusion_adaptive.pkl` | Zip `.../results/video_level_evaluation/` — **absent de votre zip actuel, à retélécharger sur Kaggle** | ⚠️ Optionnel (voir note ci-dessous) |
| `best_threshold_fusion_adaptive.npy` | Même dossier que ci-dessus | ⚠️ Optionnel |

**Si `temporal_classifier_fusion_adaptive.pkl` est absent** : le pipeline
fonctionne quand même (mesure de vitesse toujours valide), mais bascule sur
`p_video = moyenne des probabilités frame-level` avec un avertissement
explicite affiché à l'écran. **Ne citez pas un verdict obtenu dans ce mode
dégradé comme un résultat du papier** — pour cela il faut le vrai
classifieur temporel.

## 2. Installation (conda)

```bash
cd deepfake_cpu_benchmark
conda env create -f environment.yml
conda activate deepfake-cpu-bench
```

Si vous préférez réutiliser un environnement conda existant (ex. `tf210`
mentionné dans vos notes) :

```bash
conda activate tf210
pip install -r requirements.txt
```

## 3. Ajouter une vidéo de test

Déposez une ou plusieurs vidéos dans `videos_test/` (formats MP4/AVI/MOV,
comme l'app originale). Vous pouvez réutiliser une vidéo de vos jeux de
test (ex. `id59_id60_0000.mp4`, citée dans votre mémoire).

## 4. Lancer le benchmark

Mode par lot (10 frames envoyées ensemble au modèle, comme l'app d'origine) :
```bash
python -m src.benchmark --video "videos_test/id60_0006.mp4" --frames 10 --repeats 5
```

Mode "edge" strict (une frame à la fois, `batch_size=1`) :
```bash
python -m src.benchmark --video "videos_test/This is not Morgan Freeman - A Deepfake Singularity.mp4" --frames 10 --repeats 5 --single-frame
```

`--repeats 5` relance le pipeline 5 fois sur la même vidéo pour lisser la
mesure (la première exécution inclut souvent un surcoût d'initialisation
TensorFlow/CPU). Le script imprime, à la fin, un résumé qui répond
directement aux 4 questions, et sauvegarde tout dans
`results/cpu_benchmark_<date>.csv`.

## 5. Ce qui est mesuré (et comment le lire)

- **`ms_per_frame_model_forward_only`** : uniquement le passage avant dans
  le réseau (les deux branches EfficientNet-B0 + fusion + classificateur).
  C'est le chiffre directement comparable à celui du papier
  (`∼12 ms/frame sur Tesla T4`), mesuré ici sur CPU.
- **`ms_per_frame_end_to_end`** : détection de visage (SSD) + prétraitement
  + forward pass — le temps réel perçu par un utilisateur de l'app.
- **`fps_*`** : inverses des deux valeurs ci-dessus.

## 6. Limites connues de ce benchmark (à mentionner si réutilisé dans le rapport)

- Le recadrage de visage ici est un simple crop+marge+resize, **sans
  alignement par points de repère** — le mémoire mentionne des visages
  "alignés" sans préciser la méthode ; si l'app originale faisait un
  alignement par landmarks, ce script est une approximation (voir
  `src/face_detector.py`, docstring de `preprocess_for_model`). Cela n'a
  pas d'impact sur la mesure de vitesse, seulement sur le score de
  confiance affiché pour la vidéo de démonstration.
- Une seule machine testée — pas de moyenne inter-machines.
- Aucune conversion du modèle (TFLite, ONNX, quantification) : c'est du
  TensorFlow CPU "brut", exactement ce que demande la reformulation
  proposée pour répondre au Reviewer 2 (mesure honnête, sans survendre une
  optimisation edge non réalisée).

## 7. Structure du projet

```
deepfake_cpu_benchmark/
├── environment.yml
├── requirements.txt
├── README.md
├── models/                  <- vos fichiers .h5 / .pkl / .npy / .prototxt / .caffemodel
├── videos_test/              <- vos vidéos de test
├── results/                  <- CSV générés par le benchmark
└── src/
    ├── model_architecture.py # reconstruction exacte de l'architecture entraînée
    ├── face_detector.py      # détection de visage SSD (réutilisé de l'ancienne app)
    ├── temporal_features.py  # 12 features temporelles (copie exacte du notebook)
    ├── pipeline.py           # pipeline complet + mesure de temps par étape
    └── benchmark.py          # script CLI (point d'entrée)
```
conda env remove -n deepfake-cpu-bench
conda env create -f environment.yml
conda activate deepfake-cpu-bench
pip install -r requirements.txt -v
