import json
import sys
from pathlib import Path

def process_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    alignments = data.get('alignments', [])
    
    if alignments and alignments[0][0] == 'A':
        start, end = alignments[0][1]
        duration = end - start
        
        if duration > 0.2:
            new_start = end - 0.08
            alignments[0][1] = [new_start, end]
            
            with open(filepath, 'w') as f:
                json.dump(data, f)
            print(f"Updated: {filepath}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_alignments.py <directory_path>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    for json_file in directory.glob('*.json'):
        process_json_file(json_file)

if __name__ == '__main__':
    main()
