import os
import json
import torch

def load_correct_list(path, mode):
    """Load correctness list as a boolean tensor."""
    with open(path) as f:
        data = json.load(f)
    return torch.tensor(data, dtype=torch.bool)


def compare_runs(baseline_path, variant_paths, mode, save_dir=None):
    """
    Compare correctness lists for baseline vs variants.
    Saves the indices of samples where:
        - all variants correct, baseline wrong (better)
        - all variants wrong, baseline correct (worse)
    """
    baseline = load_correct_list(baseline_path, mode)
    variants = [load_correct_list(d, mode) for d in variant_paths]
    variants_tensor = torch.stack(variants)

    # Compute agreement
    all_variants_correct = variants_tensor.all(dim=0)
    all_variants_wrong = ~variants_tensor.any(dim=0)

    better = all_variants_correct & (~baseline)
    worse = all_variants_wrong & baseline

    better_indices = better.nonzero(as_tuple=True)[0].tolist()
    worse_indices = worse.nonzero(as_tuple=True)[0].tolist()

    print(f"\n===== {mode.upper()} Comparison =====")
    print(f"Better (variants all correct, baseline wrong): {len(better_indices)} samples")
    print(f"Worse (baseline correct, all variants wrong): {len(worse_indices)} samples\n")

    # Optionally save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{mode}_comparison.json")
        with open(save_path, "w") as f:
            json.dump(
                {
                    "better_indices": better_indices,
                    "worse_indices": worse_indices,
                    "baseline_path": baseline_path,
                    "variant_paths": variant_paths,
                },
                f,
                indent=2,
            )
        print(f"Saved comparison results to: {save_path}")

    # Optionally print first few indices for inspection
    if len(better_indices) > 0:
        print("Example better indices:", better_indices[:20])
    if len(worse_indices) > 0:
        print("Example worse indices:", worse_indices[:20])

    return better_indices, worse_indices


# Example usage:
if __name__ == "__main__":
    baseline_path = "comparisons/ssc_correct_old_moshi.json"
    variant_paths = [
        "comparisons/ssc_correct_5x.json",
        "comparisons/ssc_correct_text_first.json",
        "comparisons/ssc_correct_residual.json",
    ]
    save_dir = "comparisons"

    for mode in ["ssc"]:
        compare_runs(baseline_path, variant_paths, mode, save_dir)
