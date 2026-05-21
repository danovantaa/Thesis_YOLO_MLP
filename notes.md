# Definisi variabel :

- c1 = Channel Input
- c2 = Channel Output
- e = expansion ratio (yang akan dikalikan dengan output untuk hidden channel pada bottleneck)
- k = kernel
- s = stride
- p = padding
- act = Aktivasi
- c\_ = hidden channel
- g = Groups.
- d = Dilation.
- dim=1 : dalam channel

**dilation** = Memperluas area tangkapan (receptive field) tanpa menambah jumlah parameter.

**Receptive field** = area gambar yang dilihat neuron.

# Definisi variabel :

- Autopad = hitung padding otomatis agar ukuran output konvolusi = ukuran input (same padding).
- nn.Identity()= hanya meneruskan input menjadi output tanpa perubahan apa pun.

# YOLO V8 :

## Backbone

### Convolution

**Fungsi Kegunaan**

- untuk mengekstraksi fitur dari gambar

**Struktur:**

```
Conv2d -> BatchNorm2d -> SiLU
```

**Penjelasan:**

- **Conv2d** = Mengambil fitur dasar seperti garis, tepi, dan sudut.
- **BatchNorm2d** = Menormalkan nilai fitur dalam 1 batch agar training lebih stabil.
  - affine = punya parameter γ dan β (bisa belajar)
  - track_running_stats = simpan mean & var global untuk inference stabil
  - eps = nilai kecil untuk mencegah pembagian nol saat normalisasi.
  - moentum = update running mean/var secara lambat agar stabil.

- **SiLU (Sigmoid Linear Unit)** = Membuat model belajar pola non-linear dengan lebih halus dan stabil.

### Bottleneck

**Konsep**

- Mengurangi jumlah channel sebelum diproses, lalu mengembalikannya lagi, sehingga komputasi lebih ringan tapi tetap kaya fitur.

**Struktur:**

```
Convolution -> Convolution -> Shortcut
```

**Penjelasan:**

- Menggunakan dua convolution kecil untuk memproses fitur.
- Shortcut (residual connection) menjaga informasi penting agar tidak hilang.

### C2f

**Konsep**

- Mengurangi komputasi dengan memisahkan sebagian fitur, memproses subset-nya secara mendalam, lalu menggabungkannya kembali untuk menghasilkan representasi fitur yang kaya dan stabil.

**Struktur:**

```
Convolution -> Split -> Bottleneck -> Concat -> Convolution
```

**Penjelasan:**

- **Split** = Fitur dari convolution di-split menjadi 2 branch:
  1. lewat bottleneck,
  2. lewat shortcut langsung.  
     Tujuannya agar model belajar cepat dan tidak kehilangan informasi awal.

- **Bottleneck** = Fitur dipadatkan dan diproses dengan dua convolution kecil.

- **Concat** = Menggabungkan semua fitur dari bottleneck dan shortcut untuk menghasilkan representasi fitur yang lebih informatif.

### SPPF (Spatial Pyramid Pooling Fast)

**Konsep**

- Menangkap informasi konteks dari berbagai skala objek melalui pooling bertingkat secara sangat efisien, sehingga model dapat memahami objek kecil, sedang, dan besar secara bersamaan.

**Struktur :**

```
Convolution -> MaxPool2d -> MaxPool2d -> MaxPool2d -> Concat -> Convolution
```

**Penjelasan :**

- **MaxPool2d** = Mengambil nilai terbesar dari area kecil,  
  sehingga memperkuat kemampuan model dalam mendeteksi objek dengan berbagai ukuran.

## Neck

- **Upsample** = untuk menaikkan resolusi fitur (feature map)

## Head

**Struktur :**

```
Detect Head = (Conv → Conv → Conv2d(4·reg_max)) + (Conv → Conv → Conv2d(nc))
Bounding Box Loss = IoU Loss (CIoU) + Distribution Focal Loss (DFL)
Classification Loss = Binary Cross Entropy
```

### DFL

**Struktur :**

```
input  : (B, 4·reg_max, H, W)
reshape: (B, 4, reg_max, H, W)
softmax: distribusi bin
expect : Σ(p·i)
output : (B, 4, H, W)
```

**Penjelasan :** merepresentasikan jarak bounding box sebagai distribusi probabilitas lalu mengubahnya menjadi nilai jarak kontinu yang lebih presisi

# YOLO V11

## Backbone

### C3K

**Konsep**
Mirip Seperti C2F Pada Version 8, namun tidak dilakukan split menjadi banyak cabang agar menjaga efisiensi komputasi

- **Struktur:**

```
  Convolution -> Bottleneck*n -> Concat -> Convolution
```

### C3K2

**Konsep**
meningkatkan kedalaman representasi fitur tanpa menambah cabang seperti pada C2f, agar tetap efisien tapi lebih kuat dalam menangkap pola spasial.

- **Struktur:**
  ```
    Convolution -> C3K -> Concat -> Convolution # Jika c3k= True
    Convolution -> Bottleneck -> Concat -> Convolution # Jika c3k= False
  ```
- c3k=False = ekstraksi fitur sederhana dan cepat
- c3k=True = ekstraksi fitur lebih dalam dan kuat

**Penjelasan:** Fitur masukan terlebih dahulu diproses dengan convolution awal, kemudian dilewatkan ke satu blok C3K sebagai ekstraksi fitur bertingkat. Output dari blok C3K digabungkan dengan jalur shortcut melalui operasi concat untuk mempertahankan informasi awal. Selanjutnya dilakukan convolution akhir untuk menghasilkan fitur keluaran dengan dimensi.

### Attention

**Konsep :** menyoroti fitur penting secara spasial dan kanal, kemudian diperkuat dengan positional encoding depthwise convolution.

- **Struktur:**

```
Convolution -> Split (q,k) -> Sigmoid(qk) -> element-wise multiply(x, attn) -> depthwise Convolution(3×3) -> Convolution
```

**Penjelasan :** Input fitur diubah menjadi dua (q dan k). Interaksi q dan k menghasilkan peta perhatian (attention map) yang menyoroti area penting. Fitur kemudian dimodulasi oleh attention dan diperkaya dengan positional encoding melalui depthwise convolution sebelum diproyeksikan kembali ke dimensi semula.

### FFN

**Konsep :** memperluas dan mengompresi representasi kanal agar meningkatkan kapasitas non-linear setelah attention.

- **Struktur:**

```
Convolution -> Convolution
```

**Penjelasan :** memperbesar jumlah kanal untuk mempelajari kombinasi fitur yang lebih kompleks, lalu mengompres kembali ke dimensi awal. agar efisien pada fitur spasial CNN.

### PSABlock

**Konsep :** Blok attention + FFN dengan residual connection ganda untuk memperkuat representasi fitur secara efisien.

- **Struktur:**

```
Attention -> FFN
```

**Penjelasan :** menggabungkan modul Attention dan FFN dalam dua residual path berturut-turut. Attention meningkatkan fokus spasial, sedangkan FFN memperkaya representasi kanal. Residual connection menjaga stabilitas gradien dan mempertahankan informasi awal.

### C2PSA

**Konsep :** Varian C2 yang memasukkan PSABlock pada sebagian kanal untuk menggabungkan efisiensi CNN dan kemampuan global attention.

- **Struktur:**

```
Convolution -> PSABlock -> Concate -> Convolution
```

**Penjelasan :** Fitur diproyeksikan ke dimensi target, lalu dibagi menjadi dua bagian kanal. Sebagian kanal diproses oleh rangkaian PSABlock untuk menangkap hubungan global, sementara sisanya dilewatkan langsung guna menjaga efisiensi. Kedua jalur digabung kembali dan diproyeksikan untuk menghasilkan fitur akhir yang kaya konteks namun tetap ringan.

## Neck

## Head

# YOLO V12 :

## Backbone

### ABlock

**Konsep :**

- **Struktur:**

```
Attention -> Conv2d -> SiLU -> Conv2d
```

**Penjelasan :**

### A2C2F (R-ELAN)

**Konsep :**

- **Struktur:**

```
Convolution -> ABlock -> ABlock -> Concat - > Conv2d
```

**Penjelasan :**

## Neck

## Head
