import gradio as gr
import torch
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from model import NeuralGuitar
from preprocess import extract_features

# --- Configuration ---
CHECKPOINT_PATH = "checkpoints/latest.pth"
HIDDEN_SIZE = 512
N_HARMONICS = 101
N_NOISE_BANDS = 65
SAMPLE_RATE = 16000

# --- Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralGuitar(
    n_harmonics=N_HARMONICS,
    n_noise_bands=N_NOISE_BANDS,
    hidden_size=HIDDEN_SIZE
).to(device)

if os.path.exists(CHECKPOINT_PATH):
    model_file = CHECKPOINT_PATH
elif os.path.exists("latest.pth"):
    model_file = "latest.pth"
else:
    model_file = None

if model_file:
    print(f"--- Loading checkpoint: {model_file} ---")
    checkpoint = torch.load(model_file, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("--- Model loaded and ready! ---")
else:
    print(f"Warning: No checkpoint found ({CHECKPOINT_PATH} or latest.pth). Running with uninitialized weights.")

def generate_plots(f0, loudness):
    """Generates a clean visualization of Pitch and Loudness."""
    f0 = f0.squeeze().cpu().numpy()
    loudness = loudness.squeeze().cpu().numpy()
    
    # 100Hz frame rate (160 hop at 16k sr)
    time_frames = np.arange(len(f0)) * (160 / 16000)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # Dark mode/Modern aesthetic
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0b0f19')
    ax1.set_facecolor('#0b0f19')
    ax2.set_facecolor('#0b0f19')
    
    # 1. Pitch
    f0_masked = f0.copy()
    f0_masked[f0_masked <= 20] = np.nan # Hide noise/silence
    ax1.plot(time_frames, f0_masked, color='#3b82f6', linewidth=2, label='Pitch (Hz)')
    ax1.set_yscale('log')
    ax1.set_title("Pitch Trajectory (F0)", color='white', pad=10)
    ax1.set_ylabel("Frequency (Hz)", color='#9ca3af')
    ax1.grid(True, which='both', ls='--', alpha=0.1)
    ax1.tick_params(colors='#9ca3af')
    
    # 2. Loudness
    ax2.plot(time_frames, loudness, color='#ef4444', linewidth=2, label='Loudness')
    ax2.set_title("Loudness Envelope", color='white', pad=10)
    ax2.set_ylabel("Magnitude", color='#9ca3af')
    ax2.set_xlabel("Time (seconds)", color='#9ca3af')
    ax2.grid(True, which='both', ls='--', alpha=0.1)
    ax2.tick_params(colors='#9ca3af')
    
    plt.tight_layout()
    
    # Save to buffer
    plot_path = "output/feature_viz.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(plot_path, dpi=120, bbox_inches='tight', facecolor='#0b0f19')
    plt.close()
    return plot_path

def process_audio(input_audio):
    if input_audio is None:
        return None
    
    # librosa.load is more portable than torchaudio in cloud environments
    # It handles mono mixing and resampling in one step
    audio, sr = librosa.load(input_audio, sr=SAMPLE_RATE, mono=True)
    
    # Convert to torch tensor for the model
    audio = torch.from_numpy(audio).float().unsqueeze(0)
    
    # Feature Extraction
    with torch.no_grad():
        f0, loudness = extract_features(audio, SAMPLE_RATE)
        f0 = f0.to(device)
        loudness = loudness.to(device)
        
        # Synthesis
        output_audio = model(f0, loudness)
        
        # New Visualization
        plot_path = generate_plots(f0, loudness)
    
    # Save result using scipy (no backend issues)
    os.makedirs("output", exist_ok=True)
    out_path = "output/web_resynth.wav"
    
    # Convert from torch tensor to numpy for scipy
    audio_out_np = output_audio.squeeze().cpu().numpy()
    
    # Ensure it's in the right range and type for wavfile.write
    # (DDSP output is usually float32 in [-1, 1])
    wavfile.write(out_path, SAMPLE_RATE, audio_out_np)
    
    return out_path, plot_path

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>🎸 Neural Guitar: DDSP Timbre Transfer</h1>")
    gr.Markdown("""
    Convert any monophonic audio (whistling, humming, singing) into a realistic electric guitar sound!
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("🎤 Record"):
                    audio_mic = gr.Audio(source="microphone", type="filepath", label="Record your melody")
                    btn_mic = gr.Button("Generate Guitar from Recording", variant="primary")
                with gr.TabItem("📁 Upload"):
                    audio_file = gr.Audio(source="upload", type="filepath", label="Upload a .wav file")
                    btn_file = gr.Button("Generate Guitar from File", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Guitar Resynthesis")
            output_viz = gr.Image(label="Feature Visualization (Pitch & Loudness)")
            gr.Markdown("### Instructions")
            gr.Markdown("""
            1. Use one of the tabs on the left.
            2. Click the corresponding 'Generate' button.
            3. The AI will process the pitch and loudness to resynthesize it as a guitar.
            """)

    btn_mic.click(fn=process_audio, inputs=audio_mic, outputs=[output_audio, output_viz])
    btn_file.click(fn=process_audio, inputs=audio_file, outputs=[output_audio, output_viz])

if __name__ == "__main__":
    print("--- Attempting to start Gradio 3.50.2 Server ---")
    try:
        demo.launch(
            share=True,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n--- Stopping Server... ---")
    finally:
        demo.close()
        print("--- Server stopped and tunnels closed. ---")
