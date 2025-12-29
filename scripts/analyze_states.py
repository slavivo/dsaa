import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def linear_cka(X, Y, eps=1e-8):
    """
    Computes Linear CKA (Centered Kernel Alignment) between X and Y.
    X: [B, T, D1] or [N, D1]
    Y: [B, T, D2] or [N, D2]
    
    CKA is dimensionality-invariant and measures similarity between
    representational spaces rather than individual vectors.
    Range: [0, 1], where 1 = perfect alignment.
    """
    # Flatten batch and time dimensions
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    if Y.dim() == 3:
        Y = Y.reshape(-1, Y.shape[-1])
    
    X = X.float()
    Y = Y.float()

    # Center columns
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute CKA efficiently using dot products of Gram matrices
    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    
    xtx = torch.matmul(X.T, X)
    yty = torch.matmul(Y.T, Y)
    ytx = torch.matmul(Y.T, X)
    
    numerator = torch.norm(ytx, p='fro') ** 2
    denominator = torch.norm(xtx, p='fro') * torch.norm(yty, p='fro') + eps
    
    return (numerator / denominator).item()

def cosine_similarity_mean(X, Y):
    """
    Computes mean cosine similarity between X and Y across all samples.
    X, Y: [B, T, D] - must have same dimensionality D
    
    Returns mean cosine similarity across batch and time dimensions.
    """
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    if Y.dim() == 3:
        Y = Y.reshape(-1, Y.shape[-1])
    
    X = X.float()
    Y = Y.float()
    
    # Normalize
    X_norm = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity for each sample
    cos_sim = (X_norm * Y_norm).sum(dim=-1)
    
    return cos_sim.mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing 'old' and 'new' subdirectories with batch_*.pt files")
    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Directory to save plots")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity where dimensionality matches (in addition to CKA)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    old_dir = data_dir / "old"
    new_dir = data_dir / "new"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    old_files = sorted(list(old_dir.glob("batch_*.pt")))
    new_files = sorted(list(new_dir.glob("batch_*.pt")))
    
    if not old_files:
        print(f"No files found in {old_dir}")
        return
    if len(old_files) != len(new_files):
        print(f"Mismatch in .pt files between {old_dir} ({len(old_files)}) and {new_dir} ({len(new_files)})")
        return
    
    files = list(zip(old_files, new_files))

    print(f"Found {len(files)} batch pairs. Analyzing...")

    metrics = {
        # Internal New Model Metrics
        "cka_text_vs_audio": [],
        "cka_text_vs_fused": [],
        "cka_audio_vs_fused": [],
        "fusion_balance": [],  # How balanced is the fusion (0.5 = perfect balance)
        
        # Comparison with Baseline
        "cka_moshi_vs_audio": [],
        "cka_moshi_vs_fused": [],
        "cka_moshi_vs_text": [],
    }
    
    # Optional cosine similarity metrics (only when dimensions match)
    if args.use_cosine:
        metrics.update({
            "cos_text_vs_fused": [],  # If text and fused have same dim
        })

    for old_f, new_f in tqdm(files, desc="Processing batches"):
        old_data = torch.load(old_f, map_location="cpu")
        new_data = torch.load(new_f, map_location="cpu")
        
        # Ensure we have the expected keys
        if 'vanilla_qwen' in new_data and 'modified_audio' in new_data and 'modified_text_fused' in new_data and \
           'moshi_baseline' in old_data:
            
            # [B, T, D_text]
            text = new_data['vanilla_qwen']
            # [B, T, D_audio]
            audio = new_data['modified_audio']
            # [B, T, D_fused]
            fused = new_data['modified_text_fused']
            # [B, T, D_moshi]
            moshi = old_data['moshi_baseline']
            
            # Verify shapes for debugging
            if old_f == old_files[0]:  # Print only for first batch
                print(f"\nShape verification (first batch):")
                print(f"  Text (vanilla_qwen): {text.shape}")
                print(f"  Audio (modified_audio): {audio.shape}")
                print(f"  Fused (modified_text_fused): {fused.shape}")
                print(f"  Moshi (baseline): {moshi.shape}")
            
            # 1. Internal New Model Metrics (CKA - dimensionality invariant)
            metrics["cka_text_vs_audio"].append(linear_cka(text, audio))
            metrics["cka_text_vs_fused"].append(linear_cka(text, fused))
            metrics["cka_audio_vs_fused"].append(linear_cka(audio, fused))
            
            # 2. Fusion Balance: measures if fusion favors audio or text
            # Higher values = more audio-like, lower = more text-like
            cka_t_f = linear_cka(text, fused)
            cka_a_f = linear_cka(audio, fused)
            balance = cka_a_f / (cka_t_f + cka_a_f + 1e-8)
            metrics["fusion_balance"].append(balance)

            # 3. Comparison with Baseline (CKA)
            metrics["cka_moshi_vs_audio"].append(linear_cka(moshi, audio))
            metrics["cka_moshi_vs_fused"].append(linear_cka(moshi, fused))
            metrics["cka_moshi_vs_text"].append(linear_cka(moshi, text))
            
            # 4. Optional: Cosine similarity where dimensions match
            if args.use_cosine and text.shape[-1] == fused.shape[-1]:
                metrics["cos_text_vs_fused"].append(cosine_similarity_mean(text, fused))

    # --- Reporting ---
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print("\n--- Internal New Model ---")
    print(f"CKA (Text vs Audio):       {np.mean(metrics['cka_text_vs_audio']):.4f} ± {np.std(metrics['cka_text_vs_audio']):.4f}")
    print(f"  -> Low values expected (orthogonal modalities)")
    
    print(f"\nCKA (Text vs Fused):       {np.mean(metrics['cka_text_vs_fused']):.4f} ± {np.std(metrics['cka_text_vs_fused']):.4f}")
    print(f"CKA (Audio vs Fused):      {np.mean(metrics['cka_audio_vs_fused']):.4f} ± {np.std(metrics['cka_audio_vs_fused']):.4f}")
    
    print(f"\nFusion Balance:            {np.mean(metrics['fusion_balance']):.4f} ± {np.std(metrics['fusion_balance']):.4f}")
    print(f"  -> 0.5 = balanced, <0.5 = text-heavy, >0.5 = audio-heavy")
    
    if args.use_cosine and "cos_text_vs_fused" in metrics and metrics["cos_text_vs_fused"]:
        print(f"\nCosine (Text vs Fused):    {np.mean(metrics['cos_text_vs_fused']):.4f} ± {np.std(metrics['cos_text_vs_fused']):.4f}")
        print(f"  -> More interpretable alignment metric (1 = identical)")
    
    print("\n--- Baseline Comparison ---")
    print(f"CKA (Moshi vs New Audio):  {np.mean(metrics['cka_moshi_vs_audio']):.4f} ± {np.std(metrics['cka_moshi_vs_audio']):.4f}")
    print(f"  -> Does the new audio branch preserve Moshi's structure?")
    
    print(f"\nCKA (Moshi vs New Fused):  {np.mean(metrics['cka_moshi_vs_fused']):.4f} ± {np.std(metrics['cka_moshi_vs_fused']):.4f}")
    print(f"  -> How similar is the final representation to the original?")
    
    print(f"\nCKA (Moshi vs Vanilla Qwen): {np.mean(metrics['cka_moshi_vs_text']):.4f} ± {np.std(metrics['cka_moshi_vs_text']):.4f}")
    print(f"  -> Baseline similarity between Moshi and pure Text LLM.")
    
    print("="*60)

    # --- Plotting ---
    metric_keys = [
        "cka_text_vs_audio",
        "cka_text_vs_fused",
        "cka_audio_vs_fused",
        "fusion_balance",
        "cka_moshi_vs_audio",
        "cka_moshi_vs_fused",
        "cka_moshi_vs_text",
    ]
    
    metric_labels = [
        "CKA (Text vs Audio)",
        "CKA (Text vs Fused)",
        "CKA (Audio vs Fused)",
        "Fusion Balance",
        "CKA (Moshi vs Audio)",
        "CKA (Moshi vs Fused)",
        "CKA (Moshi vs Text)",
    ]
    
    if args.use_cosine and "cos_text_vs_fused" in metrics and metrics["cos_text_vs_fused"]:
        metric_keys.append("cos_text_vs_fused")
        metric_labels.append("Cosine (Text vs Fused)")

    data = [metrics[k] for k in metric_keys]

    plt.figure(figsize=(14, 8))

    plt.boxplot(
        data,
        vert=False,
        labels=metric_labels,
        showmeans=True,
        meanline=True
    )

    plt.title("Similarity Metrics Across Batches", fontsize=14)
    plt.xlabel("Similarity Value", fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    save_path = output_dir / "hidden_state_analysis_boxplots.pdf"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\nBoxplots saved to {save_path}")
    
    # Save metrics to file for later analysis
    metrics_path = output_dir / "metrics.pt"
    torch.save(metrics, metrics_path)
    print(f"Raw metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()