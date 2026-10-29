"""MobileNetV2 transfer-learning baseline: one CNN looks at the whole image and says "using phone" or "not using".

Needs TensorFlow, which has its own environment (`.venv-tf`, see the README). Run from the repo root:
    .venv-tf\\Scripts\\python.exe ml\\mobilenet_baseline.py        (add --smoke for a one-minute test run)

Trains on the TUNE split of label.csv (internal 80/20 train/validation, stratified, seed 42) and predicts the
HELD-OUT split once. Stage 1 trains a new head on the frozen ImageNet backbone; stage 2 fine-tunes its top layers.
Writes models/mobilenet_baseline.keras and, in results/mobilenet/: history.json, training_curves.png, metrics.json and
predictions_held_out.csv. ml/evaluate.py adds these predictions to the classification table as C-mobilenet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import splits  # noqa: E402
from ml.curves import plot_curves  # noqa: E402
from ml.usage_metrics import confusion, metrics  # noqa: E402

IMAGES = ROOT / "data/fpi_yolo/images"
MODEL_PATH = ROOT / "models/mobilenet_baseline.keras"
OUT = ROOT / "results/mobilenet"
SEED = 42
SIZE = 224
BATCH = 32
VAL_FRACTION = 0.20
STAGE1 = {"lr": 1e-3, "epochs": 15}
STAGE2 = {"lr": 1e-5, "epochs": 10, "unfrozen_layers": 30}
PATIENCE = 3
THRESHOLD_KEEP_WITHIN = 0.02  # keep 0.5 unless the best threshold is better by more than this much accuracy
IMBALANCE_WEIGHT_FROM = 1.2   # use class weights when the larger class is at least this many times the smaller


def image_paths(table: pd.DataFrame) -> list[str]:
    return [str(IMAGES / r.split / r.image_name) for r in table.itertuples()]


def make_dataset(paths: list[str], labels: np.ndarray, training: bool) -> tf.data.Dataset:
    """Whole image shrunk to 224 x 224 (aspect ratio not kept), pixel values 0-255; cached after the resize."""
    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (SIZE, SIZE))
        return tf.cast(tf.round(img), tf.uint8), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype("float32"))).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH).map(lambda x, y: (tf.cast(x, tf.float32), y)).prefetch(tf.data.AUTOTUNE)


def build_model() -> tuple[keras.Model, keras.Model]:
    """Augmentation -> MobileNetV2 preprocessing -> frozen backbone -> pooling -> dropout -> one sigmoid unit."""
    backbone = keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False, weights="imagenet")
    backbone.trainable = False
    inputs = keras.Input((SIZE, SIZE, 3))
    x = keras.layers.RandomFlip("horizontal")(inputs)
    x = keras.layers.RandomRotation(5 / 360)(x)  # the factor is a fraction of a full turn: +-5 degrees
    x = keras.layers.RandomBrightness(0.2)(x)
    x = keras.layers.RandomContrast(0.2)(x)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x, training=False)  # keeps the batch-norm layers in inference mode, also during fine-tuning
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs), backbone


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])


def fit_stage(model, train, val, stage: dict, class_weight) -> dict[str, list[float]]:
    compile_model(model, stage["lr"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(train, validation_data=val, epochs=stage["epochs"], callbacks=[stop],
                        class_weight=class_weight, verbose=2)
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """Best accuracy on the internal validation split; 0.5 is kept when it is within THRESHOLD_KEEP_WITHIN of the best."""
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    acc = {float(t): float(((prob >= t).astype(int) == y_true).mean()) for t in grid}
    at_half = float(((prob >= 0.5).astype(int) == y_true).mean())
    best = max(acc, key=lambda t: (acc[t], -abs(t - 0.5)))
    chosen = 0.5 if acc[best] - at_half <= THRESHOLD_KEEP_WITHIN else best
    return {"threshold": chosen, "val_accuracy_at_0.5": at_half, "best_threshold": best, "val_accuracy_at_best": acc[best]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="tiny run on 96 images, 1 epoch per stage, writes nothing")
    smoke = ap.parse_args().smoke
    keras.utils.set_random_seed(SEED)

    tune = splits.load_split("tune")
    held_out = splits.load_split("held_out")
    if smoke:
        tune, held_out = tune.sample(96, random_state=SEED), held_out.sample(32, random_state=SEED)
    train_tbl, val_tbl = train_test_split(tune, test_size=VAL_FRACTION, stratify=tune["using_phone"], random_state=SEED)
    y_train, y_val = train_tbl["using_phone"].to_numpy(), val_tbl["using_phone"].to_numpy()
    train = make_dataset(image_paths(train_tbl), y_train, training=True)
    val = make_dataset(image_paths(val_tbl), y_val, training=False)

    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    ratio = max(n_pos, n_neg) / min(n_pos, n_neg)
    weighted = ratio >= IMBALANCE_WEIGHT_FROM
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)} if weighted else None
    print(f"train {len(y_train)} (using {n_pos}, not using {n_neg}, ratio {ratio:.2f}:1, class weights {class_weight}); "
          f"validation {len(y_val)}")

    stage1, stage2 = dict(STAGE1), dict(STAGE2)
    if smoke:
        stage1["epochs"] = stage2["epochs"] = 1
    model, backbone = build_model()
    hist1 = fit_stage(model, train, val, stage1, class_weight)
    backbone.trainable = True
    for layer in backbone.layers[:-stage2["unfrozen_layers"]]:
        layer.trainable = False
    hist2 = fit_stage(model, train, val, stage2, class_weight)
    history = {k: hist1[k] + hist2[k] for k in hist1}

    prob_val = model.predict(val, verbose=0).ravel()
    choice = choose_threshold(y_val, prob_val)
    held = make_dataset(image_paths(held_out), held_out["using_phone"].to_numpy(), training=False)
    prob = model.predict(held, verbose=0).ravel()
    pred = (prob >= choice["threshold"]).astype(int)
    result = metrics(confusion(held_out["using_phone"].tolist(), pred.tolist()))
    print("threshold:", choice)
    print("held-out:", {k: round(v, 4) for k, v in result.items()})
    if smoke:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    (OUT / "history.json").write_text(json.dumps({"stage1_epochs": len(hist1["loss"]), "stage2_epochs": len(hist2["loss"]),
                                                  **history}, indent=2))
    plot_curves(history, len(hist1["loss"]), OUT / "training_curves.png")
    pd.DataFrame({"image_name": held_out["image_name"], "prob": prob, "pred": pred}).to_csv(
        OUT / "predictions_held_out.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps({
        "train_images": len(y_train), "val_images": len(y_val), "held_out_images": len(held_out),
        "class_ratio": ratio, "class_weights": class_weight, "stage1": stage1, "stage2": stage2,
        "stage1_epochs_run": len(hist1["loss"]), "stage2_epochs_run": len(hist2["loss"]), **choice, "held_out": result,
        "tensorflow": tf.__version__, "keras": keras.__version__}, indent=2))
    print(f"wrote {MODEL_PATH} and {OUT}")


if __name__ == "__main__":
    main()
