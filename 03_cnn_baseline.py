"""
STAGE 3: CNN MODEL - BASELINE (Scenario 1A architecture)
==========================================================
Follows directly from 02_segmentation_features.py output.

Implements the baseline CNN architecture from the paper (Fig. 10 / Fig. 11a,
Table 3 - "Scenario 1A"):

    Input (260, 1)
      -> Conv1D(32, kernel=3) + ReLU + BatchNorm
      -> Conv1D(64, kernel=3) + ReLU + BatchNorm
      -> Conv1D(128, kernel=5)
      -> MaxPooling1D
      -> Dropout(0.5)
      -> Flatten
      -> Dense(512)
      -> Dense(1024)
      -> Dense(6, softmax)          # 6 classes: N, L, V, R, U, A

Baseline hyperparameters (Table 3, Scenario 1A):
    learning_rate = 0.01
    batch_size    = 10
    dropout       = 0.5
    epochs        = 20
    activation    = ReLU

This script is deliberately structured so the CNN-building function
(`build_cnn_model`) and the training function (`train_and_evaluate`) can be
reused later by the RL/Q-learning tuner in Stage 4 - it just needs to call
build_cnn_model() with different hyperparameter values each iteration.

Usage:
    python 03_cnn_baseline.py --features_path ./features/100_processed_features.npz --out_dir ./results
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, log_loss

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils

# The 6 classes retained by the paper, in a fixed order for label encoding
CLASSES = ['N', 'L', 'V', 'R', 'U', 'A']
BEAT_LEN = 260  # from Stage 2 (99 pre + 160 post + 1 R-peak sample)


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
def load_all_datasets(features_paths):
    all_beats = []
    all_labels = []
    
    for path in features_paths:
        data = np.load(path, allow_pickle=True)
        if len(data['beats']) > 0:
            all_beats.append(data['beats'])
            all_labels.append(data['labels'])
            
    beats = np.concatenate(all_beats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Encode labels -> integers -> one-hot
    le = LabelEncoder()
    le.fit(CLASSES)
    y_int = le.transform(labels)
    y_onehot = utils.to_categorical(y_int, num_classes=len(CLASSES))

    # Reshape beats for Conv1D: (N, 260, 1)
    X = beats.reshape(beats.shape[0], BEAT_LEN, 1).astype(np.float32)

    print(f"Combined dataset: X={X.shape}, y={y_onehot.shape}")
    print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

    return X, y_onehot, le


# ---------------------------------------------------------------------------
# 2. BUILD CNN MODEL  (reusable — this is what the RL tuner will call later)
# ---------------------------------------------------------------------------
def build_cnn_model(input_len=BEAT_LEN,
                     num_classes=len(CLASSES),
                     filters=(32, 64, 128),
                     kernels=(3, 3, 5),
                     dropout=0.5,
                     dense_units=(512, 1024),
                     activation='relu',
                     learning_rate=0.01):
    """
    Builds the CNN architecture shown in Fig. 10/11(a) of the paper.
    All structural choices are exposed as arguments so the RL module
    (Stage 4) can vary them per Table 2's hyperparameter/action space.
    """
    model = models.Sequential(name="ECG_CNN")
    model.add(layers.Input(shape=(input_len, 1)))

    model.add(layers.Conv1D(filters[0], kernels[0], activation=activation, padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D(filters[1], kernels[1], activation=activation, padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D(filters[2], kernels[2], activation=activation, padding='same'))

    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(dropout))
    model.add(layers.Flatten())

    model.add(layers.Dense(dense_units[0], activation=activation))
    model.add(layers.Dense(dense_units[1], activation=activation))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ---------------------------------------------------------------------------
# 3. TRAIN + EVALUATE  (reusable — RL tuner will call this per iteration)
# ---------------------------------------------------------------------------
def train_and_evaluate(model, X_train, y_train, X_val, y_val,
                        batch_size=10, epochs=20, verbose=1):
    """
    Trains the given model and returns the metrics the paper's reward
    function needs: accuracy, execution (training) time, and cross-entropy
    (log loss).
    """
    t_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )
    t_end = time.time()
    train_time_min = (t_end - t_start) / 60.0

    # Predictions on the validation/test set
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_val, axis=1)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    cross_entropy = log_loss(y_val, y_pred_prob, labels=list(range(len(CLASSES))))

    metrics = {
        'accuracy': val_acc,
        'cross_entropy': cross_entropy,
        'execution_time_min': train_time_min,
        'y_true': y_true,
        'y_pred': y_pred,
        'history': history.history,
    }
    return metrics


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(features_paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load all records into one dataset
    X, y, label_encoder = load_all_datasets(features_paths)

    # 2. Train/test split (80/20, stratified to preserve class balance)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=np.argmax(y, axis=1)
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # 3. Build baseline model (Scenario 1A hyperparameters, Table 3)
    model = build_cnn_model(
        filters=(32, 64, 128),
        kernels=(3, 3, 5),
        dropout=0.5,
        dense_units=(512, 1024),
        activation='relu',
        learning_rate=0.01,
    )
    model.summary()

    # 4. Train + evaluate
    metrics = train_and_evaluate(
        model, X_train, y_train, X_val, y_val,
        batch_size=10, epochs=20,
    )

    print("\n===== BASELINE RESULTS (Scenario 1A) =====")
    print(f"Accuracy      : {metrics['accuracy']*100:.2f} %")
    print(f"Cross-entropy : {metrics['cross_entropy']:.4f}")
    print(f"Exec. time    : {metrics['execution_time_min']:.2f} min")

    # 5. Confusion matrix + classification report
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    report = classification_report(
        metrics['y_true'], metrics['y_pred'],
        target_names=label_encoder.classes_,
    )
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    # 6. Save model + results
    model.save(os.path.join(out_dir, "baseline_cnn.keras"))
    np.savez(
        os.path.join(out_dir, "baseline_results.npz"),
        confusion_matrix=cm,
        accuracy=metrics['accuracy'],
        cross_entropy=metrics['cross_entropy'],
        execution_time_min=metrics['execution_time_min'],
        classes=label_encoder.classes_,
    )
    pd.DataFrame(metrics['history']).to_csv(
        os.path.join(out_dir, "training_history.csv"), index=False
    )
    print(f"\nSaved model + results -> {out_dir}")


if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", default=None,
                         help="Path to a single *_features.npz from stage 2. "
                              "If omitted, all files in --features_dir are used.")
    parser.add_argument("--features_dir", default="./features",
                         help="Directory with *_features.npz files (default: %(default)s)")
    parser.add_argument("--out_dir", default="./results")
    args = parser.parse_args()

    if args.features_path:
        features_paths = [args.features_path]
    else:
        features_paths = sorted(glob.glob(os.path.join(args.features_dir, "*_features.npz")))
        if not features_paths:
            print(f"ERROR: No *_features.npz files found in {args.features_dir}")
            exit(1)
        print(f"Found {len(features_paths)} feature files in {args.features_dir}\n")

    main(features_paths, args.out_dir)

