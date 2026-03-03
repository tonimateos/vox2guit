import torch
import glob
import os
import argparse
from tqdm import tqdm

def check_file(fpath):
    """Checks a single .pt file for NaNs or Infs."""
    try:
        data = torch.load(fpath, map_location='cpu')
        results = {}
        for key in ['f0', 'loudness', 'audio']:
            if key in data:
                val = data[key]
                has_nan = torch.isnan(val).any().item()
                has_inf = torch.isinf(val).any().item()
                if has_nan or has_inf:
                    results[key] = {"nan": has_nan, "inf": has_inf}
        return results
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Check preprocessed dataset for NaNs and Infs.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory to scan")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data_dir, "**/*.pt"), recursive=True)
    if not files:
        print(f"No .pt files found in {args.data_dir}")
        return

    print(f"Scanning {len(files)} files in {args.data_dir}...")
    
    corrupted_files = []
    
    for f in tqdm(files):
        issue = check_file(f)
        if issue:
            corrupted_files.append((f, issue))

    print("\n" + "="*50)
    print(f"Scan complete. Total files: {len(files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print("="*50)

    if corrupted_files:
        print("\nDetails of corrupted files:")
        # Group by directory for better visibility
        by_dir = {}
        for f, issue in corrupted_files:
            d = os.path.dirname(f)
            if d not in by_dir:
                by_dir[d] = 0
            by_dir[d] += 1
            
        for d, count in by_dir.items():
            print(f"- {d}: {count} bad files")
            
        print("\nExample issues (first 5):")
        for f, issue in corrupted_files[:5]:
            print(f"  {os.path.basename(f)}: {issue}")
    else:
        print("\nNo NaNs or Infs detected in the dataset. All good!")

if __name__ == "__main__":
    main()
