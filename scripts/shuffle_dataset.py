import argparse
import json
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Shuffle dataset files except the first 6 samples."
    )
    parser.add_argument(
        "-d", "--dir", required=True, type=Path,
        help="Path to dataset directory containing extra.txt, data_stereo/, and data_text/"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    root = args.dir
    stereo_dir = root / "data_stereo"
    text_dir = root / "data_text"
    extra_path = root / "extra.txt"

    assert stereo_dir.exists(), f"{stereo_dir} not found"
    assert text_dir.exists(), f"{text_dir} not found"
    assert extra_path.exists(), f"{extra_path} not found"

    # --- Load extra.txt ---
    with open(extra_path, "r") as f:
        extra_entries = [json.loads(line) for line in f if line.strip()]

    N = len(extra_entries)
    print(f"Found {N} samples.")

    if N <= 6:
        print("Nothing to shuffle (N <= 6). Exiting.")
        return

    # --- Create shuffle mapping ---
    # First four choices, then two prepends and then 5-shot for each subject
    keep = list(range(6 + (5 * 57))) 
    shuffle_part = list(range(6 + (5 * 57), N))
    random.seed(args.seed)
    random.shuffle(shuffle_part)

    new_order = keep + shuffle_part

    # --- Rename files in place ---
    rename_map = {}
    for new_idx, old_idx in enumerate(new_order):
        rename_map[old_idx] = new_idx

    # Temporary renaming to avoid overwriting
    for old_idx, new_idx in rename_map.items():
        # data_stereo
        for ext in [".wav", ".json"]:
            src = stereo_dir / f"{old_idx}{ext}"
            tmp = stereo_dir / f"tmp_{new_idx}{ext}"
            if not src.exists():
                raise FileNotFoundError(f"Missing {src}")
            src.rename(tmp)

        # data_text
        src_txt = text_dir / f"{old_idx}.txt"
        tmp_txt = text_dir / f"tmp_{new_idx}.txt"
        if not src_txt.exists():
            raise FileNotFoundError(f"Missing {src_txt}")
        src_txt.rename(tmp_txt)

    # Final renaming to target names
    for new_idx in range(N):
        # data_stereo
        for ext in [".wav", ".json"]:
            tmp = stereo_dir / f"tmp_{new_idx}{ext}"
            dst = stereo_dir / f"{new_idx}{ext}"
            tmp.rename(dst)

        # data_text
        tmp_txt = text_dir / f"tmp_{new_idx}.txt"
        dst_txt = text_dir / f"{new_idx}.txt"
        tmp_txt.rename(dst_txt)

    # --- Update extra.txt ---
    new_extra = []
    for new_idx, old_idx in enumerate(new_order):
        entry = extra_entries[old_idx].copy()
        entry["file_index"] = new_idx
        new_extra.append(entry)

    with open(extra_path, "w") as f:
        for e in new_extra:
            f.write(json.dumps(e) + "\n")

    print("Shuffling completed successfully.")

if __name__ == "__main__":
    main()
