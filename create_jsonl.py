import sphn
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, required=True, help="Path to the dataset dir")
args = parser.parse_args()

dataset_dir = Path(args.d)

paths = [str(f) for f in (dataset_dir / "data_stereo").glob("*.wav")]
paths.sort(key=lambda x: int(Path(x).stem))
durations = sphn.durations(paths)
paths = [s.replace(f"{args.d}/", "", 1) for s in paths]

with open(dataset_dir / "data.jsonl", "w") as fobj:
    for p, d in zip(paths, durations):
        if d is None:
            continue
        json.dump({"path": p, "duration": d}, fobj)
        fobj.write("\n")