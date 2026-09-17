"""
Single Layer Perceptron - Binary Classification Iris (Setosa vs Versicolor)
Assignment 1 - Deep Learning

Implementasi ini mereplikasi spreadsheet SLP secara persis:
  - aktivasi sigmoid                g(z) = 1 / (1 + e^-z)
  - loss                           MSE = mean((g(z) - target)^2)
  - turunan                        dL/dw_j = 2 * (g - t) * (1 - g) * g * x_j
  - update stochastic per sampel   w_j <- w_j - lr * dL/dw_j
  - bobot awal 0.5, learning rate 0.1, 5 epoch

Train loss/accuracy dihitung secara berjalan (running) selama epoch, sama
seperti kolom R dan P di spreadsheet. Validation dihitung di akhir epoch
dengan bobot yang sudah dibekukan, sama seperti sheet "Validasi".

Jalankan:
    python slp_iris.py                 # varian default
    python slp_iris.py --all           # jalankan 4 varian sekaligus
"""

import argparse
import csv
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Konfigurasi (samakan dengan spreadsheet)
# --------------------------------------------------------------------------
LEARNING_RATE = 0.1
INITIAL_WEIGHT = 0.5
EPOCHS = 5
SEED = 42

# Pembagian data mengikuti template dosen: 80 training, 20 validasi.
# Baris CSV 1-40 dan 51-90 untuk training, 41-50 dan 91-100 untuk validasi.
TRAIN_IDX = list(range(0, 40)) + list(range(50, 90))
VAL_IDX = list(range(40, 50)) + list(range(90, 100))

LABEL_MAP = {"Iris-setosa": 0, "Iris-versicolor": 1}


def load_dataset(path, x4_mode):
    """Baca iris.csv dan kembalikan (train, val) sebagai list [x1, x2, x3, x4, target]."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def build(idx):
        out = []
        for i in idx:
            r = rows[i]
            x4 = float(r["petal_width"]) if x4_mode == "real" else 0.2
            out.append([
                float(r["sepal_length"]),
                float(r["sepal_width"]),
                float(r["petal_length"]),
                x4,
                LABEL_MAP[r["species"]],
            ])
        return out

    return build(TRAIN_IDX), build(VAL_IDX)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def forward(weights, sample):
    """weights = [bias, teta1..teta4]. Kembalikan g(z)."""
    bias, *teta = weights
    z = bias + sum(t * x for t, x in zip(teta, sample[:4]))
    return sigmoid(z)


def evaluate(weights, dataset):
    """Hitung MSE dan accuracy dengan bobot beku. Dipakai untuk data validasi."""
    sq_error = 0.0
    correct = 0
    for sample in dataset:
        target = sample[4]
        g = forward(weights, sample)
        sq_error += (g - target) ** 2
        correct += int((1 if g > 0.5 else 0) == target)
    n = len(dataset)
    return sq_error / n, correct / n


def train(train_set, val_set, shuffle):
    """Latih SLP dan kembalikan riwayat metrik per epoch."""
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

            # dL/dw = 2 * (g - t) * (1 - g) * g * x   (x = 1 untuk bias)
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


def plot_history(history, label, suffix):
    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Training")
    plt.plot(epochs, history["val_acc"], marker="s", label="Validation")
    plt.title(f"Accuracy per Epoch\n{label}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"accuracy_{suffix}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation")
    plt.title(f"Loss (MSE) per Epoch\n{label}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.xticks(list(epochs))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_{suffix}.png", dpi=150)
    plt.close()


def print_history(history, weights, label):
    print(f"\n{label}")
    print("-" * 66)
    print(f"{'Epoch':>5} {'TrainLoss':>11} {'ValLoss':>11} {'TrainAcc':>10} {'ValAcc':>9}")
    for i in range(EPOCHS):
        print(f"{i + 1:>5} "
              f"{history['train_loss'][i]:>11.6f} "
              f"{history['val_loss'][i]:>11.6f} "
              f"{history['train_acc'][i]:>9.2%} "
              f"{history['val_acc'][i]:>9.2%}")
    names = ["bias", "teta1", "teta2", "teta3", "teta4"]
    print("Bobot akhir: " + ", ".join(f"{n}={w:.6f}" for n, w in zip(names, weights)))


VARIANTS = [
    ("A", "template", False, "X4=0.2, urutan asli"),
    ("B", "template", True, "X4=0.2, urutan diacak"),
    ("C", "real", False, "X4 asli, urutan asli"),
    ("D", "real", True, "X4 asli, urutan diacak"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="iris.csv")
    parser.add_argument("--x4", choices=["template", "real"], default="template",
                        help="template = X4 dikunci 0.2 seperti spreadsheet dosen")
    parser.add_argument("--shuffle", action="store_true",
                        help="acak urutan data training (permutasi tetap, seed 42)")
    parser.add_argument("--all", action="store_true",
                        help="jalankan keempat varian sekaligus")
    args = parser.parse_args()

    if args.all:
        combos = VARIANTS
    else:
        combos = [("", args.x4, args.shuffle, f"X4={args.x4}, shuffle={args.shuffle}")]

    for tag, x4_mode, shuffle, desc in combos:
        train_set, val_set = load_dataset(args.data, x4_mode)
        weights, history = train(train_set, val_set, shuffle)
        label = f"Varian {tag} - {desc}" if tag else desc
        print_history(history, weights, label)
        plot_history(history, label, tag.lower() if tag else "run")

    print("\nChart tersimpan sebagai accuracy_*.png dan loss_*.png")


if __name__ == "__main__":
    main()
