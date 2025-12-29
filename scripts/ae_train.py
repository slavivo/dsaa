import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from glob import glob
from tqdm import tqdm

from compres_sae import CompresSAE  

class VectorDataset(Dataset):
    def __init__(self, data_dir: str):
        self.paths = sorted(glob(os.path.join(data_dir, "*.pt")))
        assert len(self.paths) > 0, f"No .pt files found in {data_dir}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # each file: list of 9 tensors, each shaped (B, T, D)
        tensors = torch.load(self.paths[idx])
        return tensors


def collate_batch(batch):
    # batch: list of lists (num_files × 9 tensors)
    num_streams = len(batch[0])
    out = []
    for i in range(num_streams):
        # concatenate along batch dimension
        tensors_i = [sample[i].reshape(-1, sample[i].shape[-1]) for sample in batch]
        out.append(torch.cat(tensors_i, dim=0))
    return out


def train_sparse_autoencoders(
    data_dir="saved_vectors",
    embedding_dims=[2048, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],  
    ks=[16, 8, 8, 8, 8, 8, 8, 8, 8],
    batch_size=4,
    num_epochs=6,
    lr=1e-3,
    val_per_step=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    # Load one sample to infer dimensions
    sample = torch.load(sorted(glob(os.path.join(data_dir, "*.pt")))[0])
    input_dims = [x.shape[-1] for x in sample]
    num_streams = len(input_dims)

    print(f"Found {num_streams} streams. Input dims: {input_dims}")

    # Create one SAE per stream
    models = [
        CompresSAE(input_dim=d, embedding_dim=emb_dim, k=k).to(device)
        for d, emb_dim, k in zip(input_dims, embedding_dims, ks)
    ]
    opts = [optim.Adam(m.parameters(), lr=lr) for m in models]

    # LR schedulers (OneCycleLR)
    steps_per_epoch = len(glob(os.path.join(data_dir, "*.pt"))) // batch_size
    scheds = [
        optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, epochs=num_epochs, steps_per_epoch=steps_per_epoch
        )
        for opt in opts
    ]

    # Dataset and loader
    dataset = VectorDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    # Train
    train_losses = [[] for _ in models]
    val_losses = [[] for _ in models]

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
            if (batch_idx + 1) % val_per_step == 0:
                with torch.no_grad():
                    for i, x in enumerate(batch):
                        x = x.to(device=device, dtype=torch.float32)
                        val_loss = models[i].compute_loss_dict(x)["Loss"].item()
                        val_losses[i].append(val_loss)
                        print(f"  [Step {batch_idx+1}] Model[{i}] val_loss: {val_loss:.4f}")
                continue
            
            for i, x in enumerate(batch):
                x = x.to(device=device, dtype=torch.float32)
                model, opt, sched = models[i], opts[i], scheds[i]
                losses = model.train_step(opt, x)
                sched.step()
                train_losses[i].append(losses["Loss"].item())

    plt.figure(figsize=(10, 6))
    for i in range(len(models)):
        plt.plot(train_losses[i], label=f"Model[{i}] Train Loss", alpha=0.7)
        plt.plot(
            torch.linspace(0, len(train_losses[i]), len(val_losses[i])),
            val_losses[i],
            linestyle="--",
            label=f"Model[{i}] Val Loss"
        )
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save models
    os.makedirs("checkpoints", exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f"checkpoints/sae_stream{i}.pt")
    print("✅ Training finished and models saved to checkpoints/")



if __name__ == "__main__":
    train_sparse_autoencoders()
