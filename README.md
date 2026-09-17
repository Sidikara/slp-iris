# Single Layer Perceptron — Klasifikasi Biner Iris

Assignment 1, mata kuliah Deep Learning.
Implementasi Python dari spreadsheet `PMM-TemplateSLP` untuk memisahkan
*Iris-setosa* dan *Iris-versicolor*.

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

Pembagian data mengikuti template: baris 1–40 dan 51–90 untuk training,
baris 41–50 dan 91–100 untuk validasi. Keduanya berimbang antar kelas —
40 Setosa dan 40 Versicolor untuk training, 10 dan 10 untuk validasi.

## Cara menjalankan

```bash
pip install matplotlib
python slp_iris.py
```

Menghasilkan `accuracy.png` dan `loss.png`.

## Hasil

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| 1 | 0.449889 | 0.328951 | 52.50% | 50.00% |
| 2 | 0.037452 | 0.247289 | 95.00% | 50.00% |
| 3 | 0.024372 | 0.175892 | 97.50% | 50.00% |
| 4 | 0.017357 | 0.119381 | 97.50% | 85.00% |
| 5 | 0.012740 | 0.081581 | 98.75% | 100.00% |

Bobot akhir setelah epoch 5:

```
bias  =  0.257720
teta1 = -0.241802
teta2 = -0.416903
teta3 =  1.122214
teta4 =  0.824059
```

Seluruh angka di atas identik dengan spreadsheet sampai enam angka desimal,
termasuk kelima bobot akhir.

## Dua konvensi pengukuran

Training loss dan accuracy dihitung **secara berjalan** selama epoch, dengan
bobot yang diperbarui setiap sampel. Ini mengikuti kolom `Sum Square Error` dan
`Prediksi` pada sheet SLP-Training, dan sama dengan konvensi yang dipakai
framework seperti Keras.

Validation dihitung dengan **bobot yang dibekukan di akhir epoch**, sesuai
catatan pada sheet SLP-Validation: *"in validation, bias and teta are obtained
from training for each epoch"*.

## Catatan metodologis

Validation accuracy bertahan di 50% selama tiga epoch pertama sementara training
accuracy sudah di atas 90%. Ini bukan kesalahan perhitungan.

Data training tersusun 40 Setosa lalu 40 Versicolor. Karena bobot diperbarui
setiap sampel, bobot di akhir epoch selalu terbentuk dari 40 Versicolor terakhir
sehingga condong memprediksi kelas 1. Akibatnya seluruh Setosa pada data
validasi salah diklasifikasi, tepat 10 dari 20.

Training accuracy tampak jauh lebih baik karena diukur dengan bobot yang masih
berubah di tengah epoch, bukan dengan bobot akhir.

Sebagai pembanding, mengacak urutan data training menghilangkan efek tersebut:

```bash
python slp_iris.py --shuffle
```

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| 1 | 0.443062 | 0.243234 | 53.75% | 50.00% |
| 2 | 0.070330 | 0.037353 | 96.25% | 100.00% |
| 3 | 0.017472 | 0.017840 | 100.00% | 100.00% |
| 4 | 0.010206 | 0.012399 | 100.00% | 100.00% |
| 5 | 0.007278 | 0.009790 | 100.00% | 100.00% |

Varian ini di luar ketentuan tugas dan disertakan hanya sebagai analisis.
Angka yang dilaporkan pada slide adalah hasil tanpa `--shuffle`.

## Berkas

| Berkas | Isi |
|---|---|
| `slp_iris.py` | Implementasi SLP dan pembuat chart |
| `iris.csv` | 100 baris data, sama dengan sheet Data pada spreadsheet |
| `accuracy.png`, `loss.png` | Chart hasil menjalankan script |
| `accuracy_shuffled.png`, `loss_shuffled.png` | Chart varian pembanding |

## Referensi

- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
  *Annals of Eugenics*, 7(2), 179–188.
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information
  storage and organization in the brain. *Psychological Review*, 65(6), 386–408.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
  representations by back-propagating errors. *Nature*, 323, 533–536.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository: Iris Data Set*.
  University of California, Irvine.
