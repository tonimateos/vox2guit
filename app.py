import gradio as gr
import torch
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from model import NeuralGuitar
from preprocess import extract_features

import json

from core import NeuralGuitarCore

# --- Initialize Core ---
# We point to standard locations for checkpoints and config
core = NeuralGuitarCore(
    checkpoint_path="checkpoints/latest.pth",
    config_path="config.json",
    config_name="tiny"
)
# Re-expose these for the UI plotting
SAMPLE_RATE = core.config["sample_rate"]
HOP_LENGTH = core.config["hop_length"]

def generate_plots(audio, f0, loudness):
    """Generates a clean visualization of Waveform, Pitch and Loudness."""
    audio = audio.squeeze().cpu().numpy()
    f0 = f0.squeeze().cpu().numpy()
    loudness = loudness.squeeze().cpu().numpy()
    
    # Time axes
    time_audio = np.arange(len(audio)) / SAMPLE_RATE
    # 100Hz frame rate (160 hop at 16k sr)
    time_frames = np.arange(len(f0)) * (HOP_LENGTH / SAMPLE_RATE)
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    
    # Dark mode/Modern aesthetic
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0b0f19')
    ax0.set_facecolor('#0b0f19')
    ax1.set_facecolor('#0b0f19')
    ax2.set_facecolor('#0b0f19')
    
    # 0. Waveform
    ax0.plot(time_audio, audio, color='#94a3b8', alpha=0.7, linewidth=1)
    ax0.set_title("Input Waveform", color='white', pad=10)
    ax0.set_ylabel("Amplitude", color='#9ca3af')
    ax0.grid(True, which='both', ls='--', alpha=0.1)
    ax0.tick_params(colors='#9ca3af')

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

def process_audio(input_path):
    if input_path is None:
        return None
    
    # Use the shared core for processing
    audio_orig, audio_resynth, f0, loudness = core.process_audio(input_path)
    
    # Core returns numpy/torch objects, now we handle UI-specific tasks:
    # 1. Visualization
    # Convert f0 and loudness back to torch for the plot function (it expects .cpu().numpy() calls)
    plot_path = generate_plots(torch.from_numpy(audio_orig), f0, loudness)
    
    # 2. Save result for Gradio
    os.makedirs("output", exist_ok=True)
    out_path = "output/web_resynth.wav"
    wavfile.write(out_path, SAMPLE_RATE, audio_resynth)
    
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
            output_viz = gr.Image(label="Feature Visualization (Waveform, Pitch & Loudness)")
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
