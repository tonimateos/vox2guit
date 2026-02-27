import os
import requests
import zipfile
import shutil
from tqdm import tqdm

# ==============================================================================
# GuitarSet Download Script (DDSP R&D)
# ==============================================================================
# This script automates the acquisition of the GuitarSet dataset.
# We focus on the 'audio_mono-mic' version which provides the highest 
# fidelity monophonic reference for timbre transfer.
# ==============================================================================

DATASET_URL = "https://zenodo.org/api/records/3371780/files/audio_mono-mic.zip/content"
DATA_DIR = "data/raw/guitarset"
ZIP_NAME = "audio_mono-mic.zip"

def download_file(url, filename):
    print(f"--- Downloading {filename} from Zenodo ---")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def setup_guitarset():
    # 1. Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    
    zip_path = os.path.join(DATA_DIR, ZIP_NAME)
    
    # 2. Download
    if not os.path.exists(zip_path):
        download_file(DATASET_URL, zip_path)
    else:
        print(f"Found existing {ZIP_NAME}, skipping download.")
        
    # 3. Extract
    print(f"--- Extracting to {DATA_DIR} ---")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract everything
        zip_ref.extractall(DATA_DIR)
    
    print("\n[SUCCESS] GuitarSet downloaded and extracted.")
    print(f"Files are located in: {DATA_DIR}")
    
    # Cleaning up zip to save space (Optional)
    # os.remove(zip_path)

if __name__ == "__main__":
    setup_guitarset()
