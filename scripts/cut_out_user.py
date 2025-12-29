import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

def process_file_pair(wav_path, json_path, output_dir):
    """Process a wav/json pair by trimming silence and adjusting timestamps."""
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    alignments = data.get('alignments', [])
    
    if not alignments:
        print(f"Warning: No alignments in {json_path}, skipping")
        return
    
    # Get the first timestamp
    first_word_start = alignments[0][1][0]
    
    # Calculate cut point (at least 0.1 seconds before first word)
    cut_point = max(0, first_word_start - 0.1)
        
    # Load audio (preserves all channels)
    audio, sample_rate = sf.read(wav_path)
    
    # Calculate sample index for cut point
    cut_sample = int(cut_point * sample_rate)
    
    # Trim audio from cut point onwards
    trimmed_audio = audio[cut_sample:]
    
    # Adjust all timestamps
    adjusted_alignments = []
    for word, timestamps, speaker in alignments:
        start, end = timestamps
        new_start = start - cut_point
        new_end = end - cut_point
        adjusted_alignments.append([word, [new_start, new_end], speaker])
    
    # Update data
    data['alignments'] = adjusted_alignments
    
    # Save new files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    new_wav_path = output_dir / wav_path.name
    new_json_path = output_dir / json_path.name
    
    # Write audio (preserves channel configuration)
    sf.write(new_wav_path, trimmed_audio, sample_rate)
    
    # Write JSON
    with open(new_json_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Processed: {wav_path.name} (cut {cut_point:.2f}s)")

def main():
    if len(sys.argv) != 3:
        print("Usage: python trim_audio.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)
    
    # Find all wav files
    wav_files = sorted(input_dir.glob('*.wav'))
    
    for wav_path in wav_files:
        json_path = wav_path.with_suffix('.json')
        
        if not json_path.exists():
            print(f"Warning: Missing {json_path.name}, skipping {wav_path.name}")
            continue
        
        process_file_pair(wav_path, json_path, output_dir)

if __name__ == '__main__':
    main()