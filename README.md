# Single Layer Perceptron — Klasifikasi Biner Iris

Assignment 1, mata kuliah Deep Learning.
Implementasi Python dari spreadsheet SLP untuk memisahkan *Iris-setosa* dan
*Iris-versicolor*.

## Spesifikasi model

| Komponen | Nilai |
|---|---|
| Arsitektur | Single layer perceptron, 4 input + 1 bias |
| Aktivasi | Sigmoid, `g(z) = 1 / (1 + e^-z)` |
| Loss | MSE, `mean((g(z) - t)^2)` |
| Gradien | `2 * (g - t) * (1 - g) * g * x` |
| Update | Stochastic gradient descent, per sampel |
| Learning rate | 0.1 |
| Bobot awal | 0.5 (bias dan seluruh teta) |
| Epoch | 5 |
| Data | 80 training, 20 validasi |

Pembagian data mengikuti template spreadsheet: baris 1–40 dan 51–90 untuk
training, baris 41–50 dan 91–100 untuk validasi. Masing-masing berimbang
antara kedua kelas.

## Cara menjalankan

```bash
pip install matplotlib
python slp_iris.py --all
```

Perintah di atas menjalankan keempat varian dan menyimpan chart sebagai
`accuracy_a.png`, `loss_a.png`, dan seterusnya.

Untuk satu varian saja:

```bash
python slp_iris.py --x4 template            # varian A
python slp_iris.py --x4 template --shuffle  # varian B
python slp_iris.py --x4 real                # varian C
python slp_iris.py --x4 real --shuffle      # varian D
```

## Empat varian

Template spreadsheet mengunci kolom X4 (*petal width*) pada nilai 0.2, padahal
sheet Data memuat nilai yang bervariasi. Data training juga tersusun urut per
kelas. Kedua hal itu dibuat sebagai opsi agar bisa dibandingkan.

| Varian | X4 | Urutan training |
|---|---|---|
| A | dikunci 0.2 | urut per kelas |
| B | dikunci 0.2 | diacak (seed 42) |
| C | nilai asli | urut per kelas |
| D | nilai asli | diacak (seed 42) |

## Hasil

**Varian A** — X4 = 0.2, urutan asli

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| 1 | 0.449439 | 0.371174 | 52.50% | 50.00% |
| 2 | 0.051185 | 0.298899 | 93.75% | 50.00% |
| 3 | 0.033209 | 0.227819 | 95.00% | 50.00% |
| 4 | 0.023076 | 0.165596 | 97.50% | 50.00% |
| 5 | 0.017130 | 0.116033 | 97.50% | 90.00% |

**Varian B** — X4 = 0.2, urutan diacak

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| 1 | 0.439429 | 0.317027 | 53.75% | 50.00% |
| 2 | 0.095481 | 0.046609 | 88.75% | 100.00% |
| 3 | 0.021461 | 0.021065 | 100.00% | 100.00% |
| 4 | 0.012264 | 0.014312 | 100.00% | 100.00% |
| 5 | 0.008680 | 0.011208 | 100.00% | 100.00% |

**Varian C** — X4 asli, urutan asli

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| 1 | 0.449889 | 0.328951 | 52.50% | 50.00% |
| 2 | 0.037452 | 0.247289 | 95.00% | 50.00% |
| 3 | 0.024372 | 0.175892 | 97.50% | 50.00% |
| 4 | 0.017357 | 0.119381 | 97.50% | 85.00% |
| 5 | 0.012740 | 0.081581 | 98.75% | 100.00% |

**Varian D** — X4 asli, urutan diacak

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| 1 | 0.443062 | 0.243234 | 53.75% | 50.00% |
| 2 | 0.070330 | 0.037353 | 96.25% | 100.00% |
| 3 | 0.017472 | 0.017840 | 100.00% | 100.00% |
| 4 | 0.010206 | 0.012399 | 100.00% | 100.00% |
| 5 | 0.007278 | 0.009790 | 100.00% | 100.00% |

## Catatan metodologis

Pada varian A dan C, validation accuracy bertahan di 50% selama beberapa epoch
sementara training accuracy sudah di atas 90%. Ini bukan kesalahan perhitungan.
Data training tersusun 40 Setosa lalu 40 Versicolor, dan karena bobot diperbarui
setiap sampel, bobot di akhir epoch selalu terbentuk dari 40 Versicolor terakhir
sehingga cenderung memprediksi kelas 1. Seluruh Setosa pada data validasi menjadi
salah, tepat 10 dari 20.

Training accuracy tampak jauh lebih baik karena dihitung secara berjalan selama
epoch, dengan bobot yang terus berubah, mengikuti kolom prediksi di spreadsheet.
Jika 80 data training dievaluasi ulang memakai bobot beku akhir epoch, angkanya
turun menjadi 50%, 50%, 50%, 57.5%, dan 83.75%.

Mengacak urutan data (varian B dan D) menghilangkan efek tersebut sepenuhnya.

## Verifikasi terhadap spreadsheet

Seluruh angka di atas identik dengan hasil spreadsheet sampai enam angka
desimal untuk keempat varian.

## Berkas

| Berkas | Isi |
|---|---|
| `slp_iris.py` | Implementasi SLP dan pembuat chart |
| `iris.csv` | 100 baris data, sama dengan sheet Data pada spreadsheet |
| `accuracy_*.png`, `loss_*.png` | Chart hasil menjalankan `--all` |

## Referensi

- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
  *Annals of Eugenics*, 7(2), 179–188.
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information
  storage and organization in the brain. *Psychological Review*, 65(6), 386–408.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
  representations by back-propagating errors. *Nature*, 323, 533–536.
- Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository: Iris Data Set*.
  University of California, Irvine.
