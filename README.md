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
2. **Distribution of D(x)** — discriminator scores on real samples (should cluster near 1).
3. **Distribution of D(G(z))** — discriminator scores on generated samples (should approach 0.5 at convergence).

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

![Generated Images at Epoch 10](epoch10_generated_images.png)

---

#### GAN Training Loss Curves

The **Discriminator loss** (blue) decreases and stabilises around ~0.68 (near log(2) ≈ 0.693, the theoretical optimum at equilibrium). The **Generator loss** (red) steadily rises as the discriminator becomes more capable, indicating the adversarial dynamic is functioning correctly.

![GAN Training Loss Curves](gan_loss_curves.png)

---

#### Discriminator Outputs: D(x), D(G(z)), and 1 − D(G(z))

- **D(x)** (blue) stays close to **0.5**, meaning the discriminator correctly identifies real images about half the time — a sign of equilibrium.
- **D(G(z))** (red) gradually drops below 0.5, meaning the discriminator is getting slightly better at spotting fakes.
- **1 − D(G(z))** (green dashed) rises above 0.5, representing the proportion the discriminator labels as fake.
- The dotted line marks the **0.5 equilibrium** target.

![Discriminator Outputs](discriminator_outputs.png)

---

#### GAN Training Stages — Distribution Evolution (Goodfellow 2014, Fig. 1)

This recreates the iconic Figure 1 from the original GAN paper showing how the generator distribution **p_g** (green) converges toward the real data distribution **p_data** (black dotted) over training:

| Panel | Stage | Observation |
|-------|-------|-------------|
| (a) Epoch 1 | Chaos — D unreliable | p_g far from p_data; D(x) highly variable |
| (b) Epoch 10 | D dominates | p_g starts shifting toward p_data; D sharpens |
| (c) Epoch 20 | G catching up | p_g overlaps more with p_data; D flattens |
| (d) Epoch 31 | Near equilibrium | p_g closely tracks p_data; D(x) ≈ 0.5 everywhere |

![GAN Training Stages Distribution Evolution](goodfellow_distribution_evolution.png)

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
2. **Generated spectrograms at epochs 1 and 10** — 2×4 grids showing learning progression.
3. **Loss curves** — D loss and G loss over 20 epochs.
4. **Discriminator output curves** — D(x) and D(G(z)) over epochs with 0.5 equilibrium line.
5. **Goodfellow Fig. 1 recreation** — KDE distributions at epochs 1, 5, 10, 20 for the audio GAN.
6. **BONUS — Audio synthesis:** Generated spectrogram → Griffin-Lim inversion → playable `.wav` file.

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

## References

- Goodfellow, I. et al. (2014). *Generative Adversarial Nets*. NeurIPS. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- TensorFlow/Keras DCGAN tutorial
- librosa documentation for audio feature extraction
