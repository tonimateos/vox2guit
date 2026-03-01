---
title: Neural Guitar DDSP
emoji: 🎸
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# Neural Guitar: DDSP Timbre Transfer

### 🧠 System Architecture

```text
    INPUTS (Log)           DECODER (Brain)                PARAMETERS
    [Batch, Time, 2]      [Batch, Time, 512]           [Batch, Time, 166]
   
    ┌──────────┐          ┌───────────────────┐        ┌───────────────────┐
    │  log_f0  │──┐       │   GRU (1 Layer)   │        │   MLP (Linear)    │
    └──────────┘  │       │   Hidden: 512     │        │  166 Neurons Out  │
                  ├───▶───┤                   ├───▶───┤                   │──┐
    ┌──────────┐  │       │   BatchFirst=True │        │  (Softmax/Sigmoid)│  │
    │ loudness │──┘       └───────────────────┘        └───────────────────┘  │
    └──────────┘                                                              │
                                                                              ▼
          ┌───────────────────────────────────────────────────────────────────┘
          │
          │             SYNTHESIZERS (Body)               AUDIO OUTPUT
          │           [Physics-Informed DSP]             [Batch, Sample]
          │
          │     (Amps)  ┌───────────────────────┐
          ├──────────▶──│ Harmonic Oscillator   │──┐
          │             └───────────────────────┘  │      ┌───────────┐
          │                                        ├──▶───│   SUM     │──▶ [ 🎸 ]
          │     (Mags)  ┌───────────────────────┐  │      └───────────┘
          └──────────▶──│ Noise Filter Bank     │──┘
                        └───────────────────────┘
```

**Model Details:**
- **Decoder**: Single-layer GRU (Gated Recurrent Unit) to capture temporal dependencies (slurs, vibrato, and decay).
- **Hidden Size**: 512 units.
- **Parameters**: 166 total (101 harmonic amplitudes + 65 noise band magnitudes).
- **Inductive Bias**: The model predicts control signals rather than raw samples, ensuring stable pitch.

## 🚀 Quick Start

This project uses a **local, isolated environment** to guarantee reproducibility. All dependencies are installed into the `./venv` directory within the project root.

### 1. Setup Environment
We provide a script to set up a Python 3.9 virtual environment and install exact dependencies.

```bash
# Run the setup script (creates ./venv and installs packages)
chmod +x setup_env.sh
./setup_env.sh
```

**Where are the dependencies?**
They are downloaded and installed locally in `venv/lib/python3.9/site-packages`. They do **not** affect your global system Python.

### 2. Verify Installation
Run the verification script to check that the Neural Guitar model and DSP components can be instantiated correctly.

```bash
# Verify the build
./venv/bin/python verify_project.py
```

## 🛠️ Data & Training Pipeline

### 1. Download Dataset (GuitarSet)
We use the **GuitarSet** dataset for training. Use our script to automate the download and extraction of the monophonic mic recordings.

```bash
./venv/bin/python download_data.py
```

### 2. Preprocessing
Extract high-precision pitch ($f_0$) via **CREPE** and A-weighted loudness. Our script includes a **Resume Capability**—if interrupted, it will skip already processed files.

```bash
./venv/bin/python preprocess.py --input_dir data/raw/guitarset --output_dir data/processed/guitarset
```

### 3. Quality Control (Visualization)
Before training, verify the extracted features. This tool saves a diagnostic plot to `data/visualization/`.

```bash
./venv/bin/python visualize_features.py --file data/processed/guitarset/00_SS3-84-Bb_comp_mic.pt
```

### 4. Training with Weights & Biases (W&B)
We use **W&B** for experiment tracking. It allows you to monitor loss and listen to audio samples in real-time.

1.  **Login**: ` ./venv/bin/python -m wandb login` (Paste your API key from [wandb.ai](https://wandb.ai/)).
2.  **Train**:
    ```bash
    ./venv/bin/python train.py --data_dir data/processed/guitarset --batch_size 16 --epochs 100
    ```

## 📈 Monitoring & Debugging

Training a neural synthesizer is an iterative process. Here is how to keep an eye on your model's progress:

### 1. Weights & Biases (Remote)
Once training starts, W&B provides a real-time dashboard at the URL printed in your terminal.
- **Loss Curves**: Monitor `train_loss`. A steady decrease indicates the model is learning the spectral features.
- **Audio Samples**: Every 5 epochs, the model uploads an audio reconstruction. Listen to these to hear the timbre evolve from noise to guitar. Specifically, the model extracts the pitch and loudness "DNA" from the target audio and passes it through the neural network to see how well the synthesizer can mimic the original.
- **Run Management**: Each execution gets a random name (e.g., `solar-wave-10`). You can delete failed/interrupted runs from the W&B project settings to keep your dashboard clean.

### 2. Local Monitoring
- **Progress Bar (`tqdm`)**: Shows instantaneous loss and processing speed (iterations per second) in your terminal.
- **Checkpoints**: High-fidelity model states are saved to `checkpoints/model_epoch_N.pth`.
- **Graceful Exit**: Hit `CTRL+C` at any time to stop training. The script will safely close the W&B connection and save the current state.

### 3. Auto-Resume
You don't need to do anything special to resume. If you stop the script and run the training command again, it will:
1. Detect `checkpoints/latest.pth`.
2. Automatically load the latest weights and optimizer state.
3. Continue training from the exact epoch where it was interrupted.

---

## 🧠 Architecture

The model follows the **Control-Synthesis** paradigm, separating the "Brain" (Neural Network) from the "Body" (DSP Synthesizer).

- **Encoder**: Extracts pitch ($f_0$) using a pre-trained **CREPE** model and A-weighted Loudness.
- **Decoder**: A **GRU** (Gated Recurrent Unit) maps these control signals to synthesizer parameters.
    - **Why GRU?**: We chose a GRU over a Transformer because it is significantly more efficient for real-time audio synthesis. GRUs have a strong "inductive bias" for sequences where the next frame depends heavily on the previous one (temporal persistence).
    - **Future-Proofing**: While the GRU is our "lean and mean" baseline, the modular design allows us to swap in a **Transformer-based decoder** if we need to model more complex, long-range musical dependencies in the future.
- **Synthesizer**:
    - **Harmonic Synthesizer**: Additive synthesis (sum of sines) for the tonal string vibration.
    - **Filtered Noise**: Subtractive synthesis (time-varying FIR filters) for pick attack and rasp.
- **Loss**: **Multi-Resolution STFT Loss** (in `loss.py`) ensures the model learns spectral details across time and frequency.

### Further Reading on GRUs
- **[Original Paper]**: [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) (Cho et al., 2014)
- **[Visual Guide]**: [Gated Recurrent Units (GRU) - Dive into Deep Learning](https://d2l.ai/chapter_recurrent-modern/gru.html) — An excellent, interactive textbook guide with clear diagrams and math.

---

## 📂 File Structure

- `model.py`: Main `NeuralGuitar` nn.Module.
- `synth.py`: Differentiable DSP modules (`HarmonicSynthesizer`, `FilteredNoiseSynthesizer`).
- `loss.py`: Perceptual loss functions.
- `preprocess.py`: Data pipeline for extracting $(f_0, Loudness)$ features (stable SOS filtering).
- `visualize_features.py`: Diagnostic tool for feature inspection.
- `train.py`: Training loop with W&B integration and checkpoint resume.
- `data.py`: `NeuralGuitarDataset` with random cropping.
- `download_data.py`: Automated GuitarSet downloader.
- `setup_env.sh`: Environment reproducibility script.
