# Neural Guitar: DDSP Timbre Transfer

A high-fidelity **Differentiable Digital Signal Processing (DDSP)** model implemented in PyTorch. This project performs timbre transfer, converting monophonic input (voice/humming) into a realistic electric guitar sound using a physics-informed synthesizer.

![DDSP System Architecture](./imgs/explanative_diagram.png)

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

## 🧠 Architecture

The model consists of a "Brain" (Encoder-Decoder) and a "Body" (Differentiable Synthesizer).

- **Encoder**: Extracts pitch ($f_0$) using a pre-trained **CREPE** model and Loudness (RMS).
- **Decoder**: A **GRU** (Gated Recurrent Unit) maps these control signals to synthesizer parameters.
- **Synthesizer**:
    - **Harmonic Synthesizer**: Additive synthesis (sum of sines) for the tonal string vibration.
    - **Filtered Noise**: Subtractive synthesis (time-varying FIR filters) for pick attack and rasp.
- **Loss**: **Multi-Resolution STFT Loss** (in `loss.py`) ensures the model learns spectral details across time and frequency.

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
