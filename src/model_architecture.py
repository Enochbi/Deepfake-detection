"""
Architecture du modele "Fusion adaptative" (sans DANN).

Copie EXACTE de la logique de construction utilisee pendant l'entrainement
(notebook fusion-adaptive-run-complet-corrige-kaggle.ipynb, Cellule 3), afin
que les poids sauvegardes (best_val_auc.weights.h5) se chargent sans
incompatibilite de forme ou de nom de couche.

Ne PAS modifier les noms de couches (name=...) : ils doivent correspondre
exactement a ceux utilises lors de l'entrainement pour que load_weights()
fonctionne.
"""
import gc
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetB0

INPUT_SHAPE = (224, 224, 3)


def fft_layer_gpu(x):
    """FFT 2D (RGB -> Gris -> FFT -> Log magnitude -> Normalisation [0,1]).

    Identique a la version corrigee utilisee en entrainement
    (fftshift avec axes=[1, 2], pas de decalage de l'axe batch).
    """
    gray = 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
    gray = tf.squeeze(gray, axis=-1)
    gray_complex = tf.cast(gray, tf.complex64)
    fft_result = tf.signal.fft2d(gray_complex)
    fft_shift = tf.signal.fftshift(fft_result, axes=[1, 2])
    magnitude = tf.abs(fft_shift)
    magnitude = tf.maximum(magnitude, 1e-10)
    log_magnitude = tf.math.log(1.0 + magnitude)
    min_val = tf.reduce_min(log_magnitude, axis=[1, 2], keepdims=True)
    max_val = tf.reduce_max(log_magnitude, axis=[1, 2], keepdims=True)
    normalized = tf.where(
        max_val - min_val > 1e-7,
        (log_magnitude - min_val) / (max_val - min_val + 1e-10),
        tf.zeros_like(log_magnitude)
    )
    normalized = tf.clip_by_value(normalized, 0.0, 1.0)
    return tf.stack([normalized, normalized, normalized], axis=-1)


def build_fusion_adaptive_model(
    input_shape=INPUT_SHAPE,
    try_load_imagenet_weights=False,
):
    """Reconstruit l'architecture exacte du modele entraine (sans DANN).

    try_load_imagenet_weights=False par defaut ici : inutile pour du
    deploiement, puisque les poids entraines (load_weights) ecrasent de
    toute facon l'initialisation. Le mettre a True ralentit juste le
    demarrage (telechargement des poids ImageNet pour rien).

    Une seule entree (image RGB [0,1], 224x224x3) : la branche
    frequentielle (FFT) est calculee EN INTERNE par le modele (couche
    Lambda), l'application n'a donc pas besoin de calculer la FFT elle-meme.
    """
    pretrained_weights = None
    if try_load_imagenet_weights:
        temp = EfficientNetB0(include_top=False, weights="imagenet",
                               input_shape=input_shape, pooling="avg")
        pretrained_weights = temp.get_weights()
        del temp
        gc.collect()

    input_img = layers.Input(shape=input_shape, name="input_image")

    # ---- Branche spatiale ----
    spatial_backbone = EfficientNetB0(
        include_top=False, weights=None, input_shape=input_shape,
        pooling="avg", name="efficientnetb0_spatial"
    )
    if pretrained_weights is not None:
        spatial_backbone.set_weights(pretrained_weights)
    spatial_norm = layers.Rescaling(255.0, name="spatial_rescale")(input_img)
    F_spatial = spatial_backbone(spatial_norm)
    F_spatial = layers.BatchNormalization(name="spatial_bn")(F_spatial)
    F_spatial = layers.Dropout(0.1, name="spatial_dropout")(F_spatial)

    # ---- Branche frequentielle ----
    freq_input = layers.Lambda(fft_layer_gpu, name="fft_transform")(input_img)
    freq_backbone = EfficientNetB0(
        include_top=False, weights=None, input_shape=input_shape,
        pooling="avg", name="efficientnetb0_frequency"
    )
    if pretrained_weights is not None:
        freq_backbone.set_weights(pretrained_weights)
    freq_norm = layers.Rescaling(255.0, name="freq_rescale")(freq_input)
    F_freq = freq_backbone(freq_norm)
    F_freq = layers.BatchNormalization(name="freq_bn")(F_freq)
    F_freq = layers.Dropout(0.1, name="freq_dropout")(F_freq)

    # ---- Fusion adaptative ----
    concat = layers.Concatenate(name="feature_concat")([F_spatial, F_freq])
    concat = layers.BatchNormalization(name="concat_bn")(concat)
    attention = layers.Dense(
        128, activation="relu", kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4), name="attention_dense"
    )(concat)
    attention = layers.BatchNormalization(name="attention_bn")(attention)
    attention = layers.Dropout(0.2, name="attention_dropout")(attention)
    fusion_weight = layers.Dense(
        1, activation="sigmoid",
        kernel_constraint=tf.keras.constraints.MinMaxNorm(0.0, 1.0),
        name="fusion_weight"
    )(attention)

    def adaptive_fusion(inputs):
        w, spatial, freq = inputs
        w = tf.clip_by_value(w, 0.01, 0.99)
        return w * spatial + (1.0 - w) * freq

    fused = layers.Lambda(adaptive_fusion, name="adaptive_fusion")(
        [fusion_weight, F_spatial, F_freq]
    )
    fused = layers.BatchNormalization(name="fused_bn")(fused)

    # ---- Classificateur ----
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1e-4), name="classifier_dense1")(fused)
    x = layers.BatchNormalization(name="classifier_bn1")(x)
    x = layers.Dropout(0.3, name="classifier_dropout1")(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1e-4), name="classifier_dense2")(x)
    x = layers.BatchNormalization(name="classifier_bn2")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=input_img, outputs=output, name="fusion_adaptive_no_dann")
    return model


def load_trained_model(weights_path):
    """Construit l'architecture puis charge les poids entraines."""
    model = build_fusion_adaptive_model(try_load_imagenet_weights=False)
    model.load_weights(weights_path)
    return model
