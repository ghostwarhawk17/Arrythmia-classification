"""
STAGE 4: ADAPTIVE DEEP REINFORCEMENT LEARNING MODULE
=======================================================
Implements Algorithm 1 ("Multi-objective Reinforcement Learning for ECG
classification") and the MDP formulation from Section 4.3 of:

  Serhani et al. (2025) - "Enhancing arrhythmia prediction through an
  adaptive deep reinforcement learning framework for ECG signal analysis"

MDP formalization used here (paper Section 4.3, 5-tuple (S, A, P, R, gamma)):
  - S: state = current selected value (index) for each hyperparameter
  - A: action = change a hyperparameter to one of its candidate values
  - P: transition probability (deterministic here - choosing an action
       always moves to the corresponding state)
  - R: reward = weighted sum of accuracy, execution time, and cross-entropy
       (paper Eq. in 4.3 / Algorithm 1 line 16), tuned so that high
       accuracy is rewarded, while high time and high loss are penalized -
       this matches the paper's stated optimization goal in Section 4:
       "An optimum model is one with high accuracy, low processing cost,
       and reduced execution time."
  - gamma: discount factor for the Bellman update

Q-learning (Section 4.4):
    Q*(s,a) = E[ r + gamma * max_a' Q*(s',a') ]

Algorithm 1 (as implemented here):
    while Reward_max < delta and iterations < max_iterations:
        for each hyperparameter hp in hpList:
            for each candidate value v in hpValues[hp]:
                build & train CNN with hp=v (other hp's at current best)
                compute reward
                update Q-table for hp using Bellman equation
                if reward > Reward_max: update best config

Usage:
    python 04_rl_optimizer.py --features_path ./features/100_processed_features.npz --out_dir ./results
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from tensorflow.keras import layers, models, optimizers, utils

CLASSES = ['N', 'L', 'V', 'R', 'U', 'A']
BEAT_LEN = 260


# ---------------------------------------------------------------------------
# 0. HYPERPARAMETER / NETWORK-STRUCTURE SEARCH SPACE  (Table 2 in the paper)
# ---------------------------------------------------------------------------
# CNN Training hyperparameters: learning rate, batch size, epochs
# CNN Network structure hyperparameters: dropout, activation function,
# number of dense hidden units
# (Number of conv layers is also varied, matching Scenario 1A -> 1B -> 2
#  progression of 9 -> 8 -> 8 layers in the paper)
HP_SPACE = {
    "learning_rate": [0.01, 0.005, 0.004, 0.002, 0.001],   # lr in {0.01 - xyz}
    "batch_size":    [32, 64, 128, 256],                    # Table 2
    "dropout":       [0.2, 0.3, 0.4, 0.5],                  # d in {20%-50%}
    "activation":    ["relu", "sigmoid", "tanh", "softplus"],
    "epochs":        [10, 15, 20],                          # nep > 0
    "dense_units_1": [256, 512, 1024],                      # nhu > 1
}

# Default / starting configuration (Scenario 1A style baseline defaults)
DEFAULT_HP = {
    "learning_rate": 0.01,
    "batch_size": 10 if 10 in HP_SPACE.get("batch_size", []) else 32,
    "dropout": 0.5,
    "activation": "relu",
    "epochs": 10,          # kept small for faster RL search iterations
    "dense_units_1": 512,
}
# Note: batch_size=10 isn't in Table 2's set {32,64,128,256}; we snap the
# default to the nearest valid value (32) so every action stays inside HP_SPACE.
if DEFAULT_HP["batch_size"] not in HP_SPACE["batch_size"]:
    DEFAULT_HP["batch_size"] = 32


# ---------------------------------------------------------------------------
# 1. LOAD DATA  (same as Stage 3)
# ---------------------------------------------------------------------------
def load_dataset(features_path):
    data = np.load(features_path, allow_pickle=True)
    beats = data['beats']
    labels = data['labels']

    le = LabelEncoder()
    le.fit(CLASSES)
    y_int = le.transform(labels)
    y_onehot = utils.to_categorical(y_int, num_classes=len(CLASSES))

    X = beats.reshape(beats.shape[0], BEAT_LEN, 1).astype(np.float32)
    return X, y_onehot, le


# ---------------------------------------------------------------------------
# 2. BUILD CNN MODEL  (structure driven by the current hyperparameter config)
# ---------------------------------------------------------------------------
def build_cnn_model(config, input_len=BEAT_LEN, num_classes=len(CLASSES)):
    model = models.Sequential(name="ECG_CNN_RL")
    model.add(layers.Input(shape=(input_len, 1)))

    model.add(layers.Conv1D(32, 3, activation=config["activation"], padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(64, 3, activation=config["activation"], padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(128, 5, activation=config["activation"], padding='same'))

    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(config["dropout"]))
    model.add(layers.Flatten())

    model.add(layers.Dense(config["dense_units_1"], activation=config["activation"]))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ---------------------------------------------------------------------------
# 3. TRAIN + EVALUATE ONE CANDIDATE CONFIGURATION
# ---------------------------------------------------------------------------
def train_and_evaluate(config, X_train, y_train, X_val, y_val, verbose=0):
    model = build_cnn_model(config)

    t_start = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        verbose=verbose,
    )
    t_end = time.time()
    train_time_min = (t_end - t_start) / 60.0

    y_pred_prob = model.predict(X_val, verbose=0)
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    cross_entropy = log_loss(y_val, y_pred_prob, labels=list(range(len(CLASSES))))

    return {
        "accuracy": float(val_acc),
        "cross_entropy": float(cross_entropy),
        "execution_time_min": float(train_time_min),
        "y_pred_prob": y_pred_prob,
    }, model


# ---------------------------------------------------------------------------
# 4. REWARD FUNCTION  (Algorithm 1, line 16: Reward = wt*T + wa*ACC + wm*MSE)
#    Implemented so it matches the paper's stated goal: reward high
#    accuracy, penalize high time and high error. Time and cross-entropy
#    are min-max normalized against the values seen so far in this run.
# ---------------------------------------------------------------------------
class RewardNormalizer:
    """Tracks running min/max of time & cross-entropy so the reward stays
    well-scaled across iterations (the paper does not specify the exact
    normalization, so this is a reasonable, documented implementation
    choice)."""
    def __init__(self):
        self.time_min, self.time_max = np.inf, -np.inf
        self.ce_min, self.ce_max = np.inf, -np.inf

    def update(self, t, ce):
        self.time_min = min(self.time_min, t)
        self.time_max = max(self.time_max, t)
        self.ce_min = min(self.ce_min, ce)
        self.ce_max = max(self.ce_max, ce)

    def norm_time(self, t):
        if self.time_max == self.time_min:
            return 0.0
        return (t - self.time_min) / (self.time_max - self.time_min)

    def norm_ce(self, ce):
        if self.ce_max == self.ce_min:
            return 0.0
        return (ce - self.ce_min) / (self.ce_max - self.ce_min)


def compute_reward(accuracy, exec_time_min, cross_entropy, normalizer,
                    w_time=0.2, w_acc=0.6, w_mse=0.2):
    """
    reward = wa*accuracy - wt*normalized_time - wm*normalized_cross_entropy
    Weights follow the paper's 'weighted sum of all performance factors'
    (Section 4.3) with sum(weights) == 1, as required (paper: sum_j wj = 1).
    """
    assert abs((w_time + w_acc + w_mse) - 1.0) < 1e-6, "weights must sum to 1"
    normalizer.update(exec_time_min, cross_entropy)
    nt = normalizer.norm_time(exec_time_min)
    nce = normalizer.norm_ce(cross_entropy)
    reward = w_acc * accuracy - w_time * nt - w_mse * nce
    return reward


# ---------------------------------------------------------------------------
# 5. Q-LEARNING TABLES  (Fig. 9 style Q-table, one per hyperparameter)
# ---------------------------------------------------------------------------
class QTableManager:
    """One Q-table per hyperparameter, exactly like Fig. 9 in the paper
    (states = current value index, actions = candidate value index)."""
    def __init__(self, hp_space):
        self.tables = {
            hp: np.zeros((len(vals), len(vals)))
            for hp, vals in hp_space.items()
        }

    def update(self, hp, state_idx, action_idx, reward, alpha=0.5, gamma=0.9):
        q_table = self.tables[hp]
        best_next = np.max(q_table[action_idx])  # max_a' Q(s', a')
        td_target = reward + gamma * best_next
        q_table[state_idx, action_idx] += alpha * (td_target - q_table[state_idx, action_idx])

    def best_action(self, hp, state_idx):
        return int(np.argmax(self.tables[hp][state_idx]))


# ---------------------------------------------------------------------------
# 6. ALGORITHM 1 — MAIN RL OPTIMIZATION LOOP
# ---------------------------------------------------------------------------
def optimize_cnn(hp_space, default_hp, X_train, y_train, X_val, y_val,
                  delta=0.90, max_outer_iterations=3, verbose=True):
    """
    Direct implementation of Algorithm 1 "OptimizeCNN":
        INPUT : hpList, hpValues, Dtrain, Dvalid, delta
        OUTPUT: optHPList (optimized hyperparameters)
    """
    hp_names = list(hp_space.keys())
    value_index = {hp: {v: i for i, v in enumerate(vals)} for hp, vals in hp_space.items()}

    q_manager = QTableManager(hp_space)
    normalizer = RewardNormalizer()

    reward_max = 0.0
    opt_hp = dict(default_hp)
    best_metrics = None
    best_model = None

    log_rows = []
    outer_iter = 0

    while reward_max < delta and outer_iter < max_outer_iterations:
        outer_iter += 1
        if verbose:
            print(f"\n===== Outer iteration {outer_iter}/{max_outer_iterations} "
                  f"(Reward_max so far = {reward_max:.4f}) =====")

        current_hp = dict(opt_hp)  # start each sweep from the current best

        for hp in hp_names:
            state_idx = value_index[hp][current_hp[hp]]

            for v in hp_space[hp]:
                action_idx = value_index[hp][v]

                trial_hp = dict(current_hp)
                trial_hp[hp] = v

                if verbose:
                    print(f"  Trying {hp} = {v} ... ", end="", flush=True)

                metrics, model = train_and_evaluate(trial_hp, X_train, y_train, X_val, y_val)
                reward = compute_reward(
                    metrics["accuracy"], metrics["execution_time_min"], metrics["cross_entropy"],
                    normalizer,
                )

                # Bellman update for this hyperparameter's Q-table
                q_manager.update(hp, state_idx, action_idx, reward)

                log_rows.append({
                    "outer_iter": outer_iter, "hyperparameter": hp, "value": v,
                    "accuracy": metrics["accuracy"], "cross_entropy": metrics["cross_entropy"],
                    "execution_time_min": metrics["execution_time_min"], "reward": reward,
                })

                if verbose:
                    print(f"acc={metrics['accuracy']:.4f}, "
                          f"ce={metrics['cross_entropy']:.4f}, "
                          f"time={metrics['execution_time_min']:.2f}min, "
                          f"reward={reward:.4f}")

                if reward > reward_max:
                    reward_max = reward
                    opt_hp[hp] = v
                    best_metrics = metrics
                    best_model = model
                    if verbose:
                        print(f"    -> New best! {hp}={v}, reward={reward:.4f}")

            # After sweeping all values for this hyperparameter, "lock in"
            # the Q-table's preferred action (paper: highest Q-value column
            # indicates the best action for that hyperparameter, Fig. 9)
            best_action_idx = q_manager.best_action(hp, state_idx)
            opt_hp[hp] = hp_space[hp][best_action_idx]

    log_df = pd.DataFrame(log_rows)
    return opt_hp, reward_max, best_metrics, best_model, q_manager, log_df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(features_path, out_dir, delta, max_outer_iterations):
    os.makedirs(out_dir, exist_ok=True)

    X, y, label_encoder = load_dataset(features_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    opt_hp, reward_max, best_metrics, best_model, q_manager, log_df = optimize_cnn(
        HP_SPACE, DEFAULT_HP, X_train, y_train, X_val, y_val,
        delta=delta, max_outer_iterations=max_outer_iterations,
    )

    print("\n===== RL OPTIMIZATION COMPLETE =====")
    print("Optimized hyperparameters:", json.dumps(opt_hp, indent=2))
    print(f"Best reward   : {reward_max:.4f}")
    print(f"Accuracy      : {best_metrics['accuracy']*100:.2f} %")
    print(f"Cross-entropy : {best_metrics['cross_entropy']:.4f}")
    print(f"Exec. time    : {best_metrics['execution_time_min']:.2f} min")

    # Save everything Stage 5 (evaluation) needs
    y_true = np.argmax(y_val, axis=1)
    np.savez(
        os.path.join(out_dir, "rl_optimized_results.npz"),
        y_true=y_true,
        y_pred_prob=best_metrics["y_pred_prob"],
        accuracy=best_metrics["accuracy"],
        cross_entropy=best_metrics["cross_entropy"],
        execution_time_min=best_metrics["execution_time_min"],
        classes=label_encoder.classes_,
        best_hyperparameters=json.dumps(opt_hp),
    )
    best_model.save(os.path.join(out_dir, "rl_optimized_cnn.keras"))
    log_df.to_csv(os.path.join(out_dir, "rl_search_log.csv"), index=False)

    with open(os.path.join(out_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(opt_hp, f, indent=2)

    print(f"\nSaved RL-optimized model + results -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", required=True,
                         help="Path to *_features.npz from stage 2")
    parser.add_argument("--out_dir", default="./results")
    parser.add_argument("--delta", type=float, default=0.90,
                         help="Reward threshold to stop the search (Algorithm 1's delta)")
    parser.add_argument("--max_outer_iterations", type=int, default=3,
                         help="Cap on outer while-loop sweeps (safety limit; "
                              "the paper's Algorithm 1 loops until reward > delta)")
    args = parser.parse_args()

    main(args.features_path, args.out_dir, args.delta, args.max_outer_iterations)
