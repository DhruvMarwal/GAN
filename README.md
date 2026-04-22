# Lab 4 — Generative Adversarial Networks (GAN)

**Course:** Advanced Topics in Machine Learning | B.Tech/MBA Tech AI — Semester VI  
**Institution:** SVKM's NMIMS, Mukesh Patel School of Technology Management & Engineering  
**Faculty:** Dr. Ami Munshi

---

## Aim

To implement and evaluate Generative Adversarial Networks (GAN) across three distinct datasets:

- **Notebook 1 (`I043_Lab4_ATML.ipynb`)** — Text data (Daily Conversations TF-IDF vectors)  
- **Notebook 2 (`Lab4_I043.ipynb`)** — Image data (Fashion MNIST)  
- **Notebook 3 (`I043_Lab4.ipynb`)** — Audio data (Speech Commands — "dog" class mel spectrograms)

---

## Theoretical Background

### What is a GAN?

A **Generative Adversarial Network (GAN)**, introduced by Goodfellow et al. (2014), is a framework where two neural networks compete in an adversarial game:

- **Generator (G):** Takes random noise `z` as input and generates synthetic samples resembling the training distribution.
- **Discriminator (D):** Classifies inputs as real (from training data) or fake (from the generator).

The training objective is a **minimax game**:

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

- The **Discriminator** maximises the objective — tries to correctly identify real images (D(x) → 1) and fake images (D(G(z)) → 0).
- The **Generator** minimises the objective — tries to fool the discriminator (D(G(z)) → 1).

At **Nash equilibrium**, the generator produces samples indistinguishable from real data and D(x) = 0.5 for all inputs.

---

## Notebook 1 — GAN on Text (TF-IDF Vectors)

**Dataset:** `aarohanverma/simple-daily-conversations-cleaned` (Hugging Face)

### Pipeline

```python
from datasets import load_dataset
ds = load_dataset("aarohanverma/simple-daily-conversations-cleaned")
df = pd.DataFrame(ds['train'])
```

### Data Preprocessing

- Text conversations converted to **100-dimensional TF-IDF vectors** using `TfidfVectorizer(max_features=100)`.
- Converted to PyTorch tensors and batched (batch size = 32).

### Architecture

**Discriminator (PyTorch):**

```python
class Discriminator(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
```

**Generator (PyTorch):**

```python
latent_dim = 20

class Generator(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 100),   # output matches TF-IDF size
        )
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch Size | 32 |
| Latent Dimension | 20 |
| Optimizer | Adam (lr = 0.0002) |
| Loss | Binary Cross-Entropy (BCELoss) |
| Discriminator updates per G step (k) | 1 |

### Training Loop (Pseudocode)

```
For each epoch:
    For each batch:
        [Train Discriminator]
        - Compute loss on real samples (label = 1)
        - Generate fake samples from noise z ~ N(0,1)
        - Compute loss on fake samples (label = 0)
        - Backpropagate D loss

        [Train Generator]
        - Generate new fake samples
        - Compute D's output on fakes (target label = 1 to fool D)
        - Backpropagate G loss
```

### Outputs

1. **Probability distribution of real TF-IDF values** — histogram of real sample feature values.
> <img width="465" height="378" alt="image" src="https://github.com/user-attachments/assets/7f78f093-8f96-4a64-9522-a4f7029f30b9" />

2. **Distribution of D(x)** — discriminator scores on real samples (should cluster near 1).
> <img width="463" height="377" alt="image" src="https://github.com/user-attachments/assets/7fc45731-de2f-4e20-bb75-5dd55c53c403" />

3. **Distribution of D(G(z))** — discriminator scores on generated samples (should approach 0.5 at convergence).
> <img width="482" height="375" alt="image" src="https://github.com/user-attachments/assets/57fe138c-d265-4a00-be00-d6393be777f0" />

---

## Notebook 2 — GAN on Fashion MNIST

**Dataset:** `keras.datasets.fashion_mnist` — 60,000 grayscale images (28×28) across 10 clothing categories.

### Data Preprocessing

```python
x_train = (x_train.astype('float32') - 127.5) / 127.5   # Normalise to [-1, 1]
x_train = x_train.reshape(-1, 28, 28, 1)
```

Sample classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

### Architecture

**Generator (TensorFlow/Keras — DCGAN-style):**

| Layer | Details |
|-------|---------|
| Dense | 7×7×256, no bias |
| BatchNorm + LeakyReLU(0.2) | — |
| Reshape | (7, 7, 256) |
| Conv2DTranspose 128 | 5×5, stride 1, same |
| BatchNorm + LeakyReLU(0.2) | — |
| Conv2DTranspose 64 | 5×5, stride 2, same |
| BatchNorm + LeakyReLU(0.2) | — |
| Conv2DTranspose 1 | 5×5, stride 2, tanh → (28, 28, 1) |

**Discriminator (TensorFlow/Keras):**

| Layer | Details |
|-------|---------|
| Conv2D 64 | 5×5, stride 2, same, input (28,28,1) |
| LeakyReLU(0.2) + Dropout(0.3) | — |
| Conv2D 128 | 5×5, stride 2, same |
| LeakyReLU(0.2) + Dropout(0.3) | — |
| Flatten + Dense 1 | sigmoid |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 32 |
| Noise Dimension | 100 |
| Optimizer (G & D) | Adam (lr = 0.0002, β₁ = 0.5) |
| Loss | Binary Cross-Entropy |
| Label Smoothing | Real labels → 0.9, Fake labels → 0.05 |

### Results & Visualisations

#### Generated Images — Epoch 10

At epoch 10, the generator already produces recognisable fashion items — shoes, shirts, dresses, and boots — with visible structural detail, though some noise is still present.
> <img width="379" height="481" alt="image" src="https://github.com/user-attachments/assets/5710db59-0bfc-4e31-9f7e-febd448b7313" />
> <img width="365" height="478" alt="image" src="https://github.com/user-attachments/assets/abaeb849-1226-46e4-8f94-3b34af03b660" />
> <img width="369" height="417" alt="image" src="https://github.com/user-attachments/assets/cc888e94-d59c-4d20-b28c-374ce1ccb57f" />

---

#### GAN Training Loss Curves

The **Discriminator loss** (blue) decreases and stabilises around ~0.68 (near log(2) ≈ 0.693, the theoretical optimum at equilibrium). The **Generator loss** (red) steadily rises as the discriminator becomes more capable, indicating the adversarial dynamic is functioning correctly.

> <img width="828" height="363" alt="image" src="https://github.com/user-attachments/assets/caa156cf-3fb6-44ad-b715-5029364e951d" />

---

#### Discriminator Outputs: D(x), D(G(z)), and 1 − D(G(z))

- **D(x)** (blue) stays close to **0.5**, meaning the discriminator correctly identifies real images about half the time — a sign of equilibrium.
- **D(G(z))** (red) gradually drops below 0.5, meaning the discriminator is getting slightly better at spotting fakes.
- **1 − D(G(z))** (green dashed) rises above 0.5, representing the proportion the discriminator labels as fake.
- The dotted line marks the **0.5 equilibrium** target.

> <img width="833" height="460" alt="image" src="https://github.com/user-attachments/assets/5ac00f5f-8d57-4260-9bca-1a28f3692fc7" />

---

#### GAN Training Stages — Distribution Evolution (Goodfellow 2014, Fig. 1)

This recreates the iconic Figure 1 from the original GAN paper showing how the generator distribution **p_g** (green) converges toward the real data distribution **p_data** (black dotted) over training:

| Panel | Stage | Observation |
|-------|-------|-------------|
| (a) Epoch 1 | Chaos — D unreliable | p_g far from p_data; D(x) highly variable |
| (b) Epoch 10 | D dominates | p_g starts shifting toward p_data; D sharpens |
| (c) Epoch 20 | G catching up | p_g overlaps more with p_data; D flattens |
| (d) Epoch 31 | Near equilibrium | p_g closely tracks p_data; D(x) ≈ 0.5 everywhere |

> <img width="1487" height="389" alt="image" src="https://github.com/user-attachments/assets/be1574da-476b-4f09-a405-6eafc315715e" />

### Discriminator Probability Distributions After Training
> <img width="831" height="454" alt="image" src="https://github.com/user-attachments/assets/fb4aecb2-295a-416a-8ebb-f5aaf0c29382" />

---

## Notebook 3 — GAN on Audio (Speech Spectrograms)

**Dataset:** Google Speech Commands v0.02 — "dog" word class (≈2,000 WAV files).

### Data Preprocessing

```python
TARGET_WORD = "dog"
SR          = 16000     # Sample rate (Hz)
DURATION    = 1.0       # seconds
N_MELS      = 64        # Mel filter banks
HOP_LENGTH  = 256

# Pipeline per file:
# 1. Load WAV → pad/trim to 1 second
# 2. Compute Mel spectrogram (power)
# 3. Convert to dB scale (librosa.power_to_db)
# 4. Normalise dB values to [-1, 1]
# Output shape per sample: (64, 63, 1)
```

### Architecture

**Generator (TensorFlow/Keras — produces 64×63 spectrograms):**

| Layer | Details |
|-------|---------|
| Dense | 4×4×256, no bias, input = noise_dim=128 |
| BatchNorm + LeakyReLU(0.2) | — |
| Reshape | (4, 4, 256) |
| Conv2DTranspose 128 | 4×4, stride 2 → 8×8 |
| BatchNorm + LeakyReLU(0.2) | — |
| Conv2DTranspose 64 | 4×4, stride 2 → 16×16 |
| BatchNorm + LeakyReLU(0.2) | — |
| Conv2DTranspose 32 | 4×4, stride 2 → 32×32 |
| BatchNorm + LeakyReLU(0.2) | — |
| Conv2DTranspose 1 | 4×4, stride 2, tanh → 64×64 |
| Cropping2D | Crop 1 column → (64, 63, 1) |

**Discriminator (TensorFlow/Keras):**

| Layer | Details |
|-------|---------|
| Conv2D 64 | 4×4, stride 2, input (64, 63, 1) |
| LeakyReLU(0.2) + Dropout(0.3) | — |
| Conv2D 128 | 4×4, stride 2, same |
| LeakyReLU(0.2) + Dropout(0.3) | — |
| Conv2D 256 | 4×4, stride 2, same |
| LeakyReLU(0.2) + Dropout(0.3) | — |
| Flatten + Dense 1 | sigmoid |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch Size | 128 |
| Noise Dimension | 128 |
| Optimizer (G & D) | Adam (lr = 0.0002, β₁ = 0.5) |
| Loss | Binary Cross-Entropy |
| Label Smoothing | Real → 0.9, Fake → 0.05 |

### Outputs

1. **Real mel spectrograms** — 2×5 grid of actual "dog" speech spectrograms.
> <img width="1315" height="484" alt="image" src="https://github.com/user-attachments/assets/6ea4e19d-17e1-4858-b02d-b1cd62ad7b6b" />

2. **Generated spectrograms at epochs** — 2×4 grids showing learning progression.
> <img width="996" height="569" alt="image" src="https://github.com/user-attachments/assets/f2cbc22e-cc74-4aa3-9e65-30918ef88db6" />
> <img width="741" height="416" alt="image" src="https://github.com/user-attachments/assets/7561e451-8c46-471a-9ac7-b81e9e84646c" />
> <img width="747" height="303" alt="image" src="https://github.com/user-attachments/assets/e634e63e-0fb5-4cbf-896e-98e2b9d8fa6f" />

3. **Loss curves** — D loss and G loss over 20 epochs.
> <img width="747" height="322" alt="image" src="https://github.com/user-attachments/assets/f1e80a2f-8525-4d61-803f-80b67424cccd" />

5. **Discriminator output curves** — D(x) and D(G(z)) over epochs with 0.5 equilibrium line.
> <img width="752" height="318" alt="image" src="https://github.com/user-attachments/assets/52a7ec43-8984-4ac5-a201-5fb9fe3f08c5" />

7. **Goodfellow Fig. 1 recreation** — KDE distributions at epochs 1, 5, 10, 20 for the audio GAN.
> <img width="1320" height="348" alt="image" src="https://github.com/user-attachments/assets/646504ff-8645-4a7c-856c-b72c68898a09" />

```python
# Bonus: Convert generated spectrogram back to audio
gen_spec_db  = denormalise(generator(noise))
gen_mel      = librosa.db_to_power(gen_spec_db)
gen_audio    = librosa.feature.inverse.mel_to_audio(gen_mel, sr=SR, n_iter=60)
sf.write('generated_dog.wav', gen_audio, SR)
```

---

## Summary Comparison

| Aspect | Notebook 1 (Text) | Notebook 2 (Fashion MNIST) | Notebook 3 (Audio) |
|--------|-------------------|-----------------------------|---------------------|
| **Dataset** | Daily conversations (TF-IDF) | Fashion MNIST images | Speech commands ("dog") |
| **Input shape** | 100-dim vector | (28, 28, 1) image | (64, 63, 1) spectrogram |
| **Framework** | PyTorch | TensorFlow/Keras | TensorFlow/Keras |
| **Architecture** | Fully-connected MLP | DCGAN (Conv + Transposed Conv) | DCGAN (deeper, 4 upsampling stages) |
| **Noise dim** | 20 | 100 | 128 |
| **Epochs** | 20 | 50 | 20 |
| **Key extra** | — | Label smoothing, snapshots | Label smoothing, Griffin-Lim audio synthesis |

---

## Key Learnings

- GANs can be adapted to diverse data modalities — text, images, and audio — with appropriate preprocessing and architecture choices.
- **Label smoothing** (real labels → 0.9 instead of 1.0) helps stabilise training by preventing the discriminator from becoming overconfident.
- The **Goodfellow 2014 Fig. 1** (KDE of p_data, p_g, and D(x)) provides an intuitive visualisation of GAN convergence — the generator distribution gradually aligns with the real data distribution while D(x) approaches 0.5.
- For audio, GANs operate on **mel spectrogram representations** rather than raw waveforms; generated spectrograms can be inverted back to audio using Griffin-Lim.
- D loss stabilising near **0.693** (= log 2) is a strong indicator that training has reached a healthy equilibrium.

---

