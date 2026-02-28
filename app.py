import gradio as gr
import torch
import torchaudio
import os
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
    print(f"--- Loading checkpoint: {CHECKPOINT_PATH} ---")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("--- Model loaded and ready! ---")
else:
    print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found. Running with uninitialized weights.")

def process_audio(input_audio):
    if input_audio is None:
        return None
    
    # input_audio is a path to the temporary wav file
    audio, sr = torchaudio.load(input_audio)
    
    # Mix to mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        
    # Resample to 16k
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio = resampler(audio)
    
    # Feature Extraction
    with torch.no_grad():
        f0, loudness = extract_features(audio, SAMPLE_RATE)
        f0 = f0.to(device)
        loudness = loudness.to(device)
        
        # Synthesis
        output_audio = model(f0, loudness)
    
    # Save result
    os.makedirs("output", exist_ok=True)
    out_path = "output/web_resynth.wav"
    torchaudio.save(out_path, output_audio.cpu(), SAMPLE_RATE)
    
    return out_path

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
            gr.Markdown("### Instructions")
            gr.Markdown("""
            1. Use one of the tabs on the left.
            2. Click the corresponding 'Generate' button.
            3. The AI will process the pitch and loudness to resynthesize it as a guitar.
            """)

    btn_mic.click(fn=process_audio, inputs=audio_mic, outputs=output_audio)
    btn_file.click(fn=process_audio, inputs=audio_file, outputs=output_audio)

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
