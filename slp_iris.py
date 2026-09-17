"""
Single Layer Perceptron - Binary Classification Iris (Setosa vs Versicolor)
Assignment 1 - Deep Learning

Replikasi persis dari spreadsheet PMM-TemplateSLP:
  aktivasi   g(z) = 1 / (1 + e^-z)
  loss       MSE  = mean((g(z) - target)^2)
  gradien    dL/dw_j = 2 * (g - t) * (1 - g) * g * x_j
  update     stochastic gradient descent, per sampel
  bobot awal 0.5 untuk bias dan seluruh teta
  lr 0.1, 5 epoch, 80 data training, 20 data validasi

Sheet SLP-Training menghitung loss dan accuracy secara berjalan selama epoch,
dengan bobot yang terus diperbarui tiap sampel. Sheet SLP-Validation memakai
bobot yang dibekukan di akhir epoch, sesuai catatan pada template:
"in validation, bias and teta are obtained from training for each epoch".
Kode ini mengikuti kedua konvensi tersebut.

Jalankan:
    python slp_iris.py                # sesuai spreadsheet
    python slp_iris.py --shuffle      # eksperimen: urutan training diacak
"""

import argparse
import csv
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
INITIAL_WEIGHT = 0.5
EPOCHS = 5
SEED = 42

# Pembagian data mengikuti template: baris CSV 1-40 dan 51-90 untuk training,
# baris 41-50 dan 91-100 untuk validasi. Keduanya berimbang antar kelas.
TRAIN_IDX = list(range(0, 40)) + list(range(50, 90))
VAL_IDX = list(range(40, 50)) + list(range(90, 100))

LABEL_MAP = {"Iris-setosa": 0, "Iris-versicolor": 1}


def load_dataset(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def build(idx):
        return [[
            float(rows[i]["sepal_length"]),
            float(rows[i]["sepal_width"]),
            float(rows[i]["petal_length"]),
            float(rows[i]["petal_width"]),
            LABEL_MAP[rows[i]["species"]],
        ] for i in idx]

    return build(TRAIN_IDX), build(VAL_IDX)


def forward(weights, sample):
    bias, *teta = weights
    z = bias + sum(t * x for t, x in zip(teta, sample[:4]))
    return 1.0 / (1.0 + math.exp(-z))


def evaluate(weights, dataset):
    """MSE dan accuracy dengan bobot beku - dipakai untuk data validasi."""
    sq_error = 0.0
    correct = 0
    for sample in dataset:
        g = forward(weights, sample)
        sq_error += (g - sample[4]) ** 2
        correct += int((1 if g > 0.5 else 0) == sample[4])
    n = len(dataset)
    return sq_error / n, correct / n


def train(train_set, val_set, shuffle=False):
    if shuffle:
        order = list(range(len(train_set)))
        random.Random(SEED).shuffle(order)
        train_set = [train_set[i] for i in order]

    weights = [INITIAL_WEIGHT] * 5
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for _ in range(EPOCHS):
        sq_error = 0.0
        correct = 0

        for sample in train_set:
            x1, x2, x3, x4, target = sample
            g = forward(weights, sample)

            sq_error += (g - target) ** 2
            correct += int((1 if g > 0.5 else 0) == target)

            delta = 2.0 * (g - target) * (1.0 - g) * g
            for j, x in enumerate([1.0, x1, x2, x3, x4]):
                weights[j] -= LEARNING_RATE * delta * x

        n = len(train_set)
        history["train_loss"].append(sq_error / n)
        history["train_acc"].append(correct / n)

        val_loss, val_acc = evaluate(weights, val_set)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return weights, history


def plot_history(history, suffix=""):
    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Training")
    plt.plot(epochs, history["val_acc"], marker="s", label="Validation")
    plt.title("Accuracy per Epoch - Training vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"accuracy{suffix}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation")
    plt.title("Loss (MSE) per Epoch - Training vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.xticks(list(epochs))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss{suffix}.png", dpi=150)
    plt.close()


def print_history(history, weights):
    print(f"{'Epoch':>5} {'TrainLoss':>11} {'ValLoss':>11} {'TrainAcc':>10} {'ValAcc':>9}")
    for i in range(EPOCHS):
        print(f"{i + 1:>5} "
              f"{history['train_loss'][i]:>11.6f} "
              f"{history['val_loss'][i]:>11.6f} "
              f"{history['train_acc'][i]:>9.2%} "
              f"{history['val_acc'][i]:>9.2%}")
    names = ["bias", "teta1", "teta2", "teta3", "teta4"]
    print("\nBobot akhir: " + ", ".join(f"{n}={w:.6f}" for n, w in zip(names, weights)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="iris.csv")
    parser.add_argument("--shuffle", action="store_true",
                        help="eksperimen tambahan: acak urutan data training (seed 42)")
    args = parser.parse_args()

    train_set, val_set = load_dataset(args.data)
    weights, history = train(train_set, val_set, args.shuffle)

    print("Urutan training:", "diacak (seed 42)" if args.shuffle else "sesuai spreadsheet")
    print()
    print_history(history, weights)

    suffix = "_shuffled" if args.shuffle else ""
    plot_history(history, suffix)
    print(f"\nChart tersimpan: accuracy{suffix}.png, loss{suffix}.png")


if __name__ == "__main__":
    main()
