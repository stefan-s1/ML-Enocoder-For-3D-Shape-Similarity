import os, time, random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import trimesh

from glob import glob
import argparse
import logging
from tqdm import tqdm

# NOTE: B, V, D, H, W stands for Batch Size, Number of Views, Depth, Height, Width respectively 

# Since voxelisation is expensive, we will cache the results of each voxelisation of each training sample into a corresponding .voxel.pt file to save resources during training
class Voxelised_3D_Data(Dataset):
    def __init__(self, files, dimensions=64, cache_ext = ".voxel.pt"):
        self.files = files
        self.dimensions = dimensions
        self.cache_ext = cache_ext

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file       = self.files[idx]
        cache_path = file + self.cache_ext
        D          = self.dimensions

        voxel_grid = None
 
        if os.path.exists(cache_path):
            vg = torch.load(cache_path)
            # validate that cached grid is the right shape
            if isinstance(vg, torch.Tensor) and vg.shape == (D, D, D):
                voxel_grid = vg

        if voxel_grid is None:
            mesh = trimesh.load_mesh(file)
            pitch = mesh.extents.max() / D

            # initial voxelization + true cubic resampling
            vg   = mesh.voxelized(pitch)
            vg_n = vg.revoxelized((D, D, D))
            mat  = vg_n.matrix                  # numpy bool array (D,D,D)
            voxel_grid = torch.from_numpy(mat)  # tensor (D,D,D)
            torch.save(voxel_grid, cache_path)

        # Now voxel_grid is guaranteed to be (D,D,D)
        augs = self.augment_voxel_grid(voxel_grid.numpy())
        augs = np.stack(augs, axis=0).astype(np.float32)  # shape (V, D, D, D)

        return torch.from_numpy(augs)  # Tensor(V, D, D, D)

    def augment_voxel_grid(
        self,
        grid: np.ndarray,
        n_augs: int = 4,              
        drop_rate: float = 0.10,
        occlude_rate: float = 0.4,
        occlude_size_frac: float = 0.2,# up to 20% volume
        max_shift: int = 5,           
        max_crop_frac: float = 0.15
    ) -> list[np.ndarray]:
        """
        Generate a set of augmented voxel grids (positive views) including:
        - Random 90° rotations and flips
        - Random voxel dropout
        - Random cuboid occlusion
        - Random translation (roll)
        - Random crop-and-pad
        Returns List of numpy boolean arrays shape (n,n,n)
        """
        assert grid.ndim == 3 and grid.dtype == bool
        n = grid.shape[0]
        augs = [grid]

        for _ in range(n_augs):
            g = grid.copy()

            # 1) random small-angle rotation (<15 degrees) - not working yet
            """
            if random.random() < 0.5:
                angle = random.uniform(-15, 15) * np.pi/180
                axis = random.choice([(1,2), (0,2), (0,1)])
                g = np.rot90(g, k=1, axes=axis) if abs(angle) > (np.pi/4) else g  # fallback to 90°
            """

            # 2) 90° rotations & flips
            axes = random.choice([(0,1), (0,2), (1,2)])
            k = random.randint(0, 3)
            g = np.rot90(g, k=k, axes=axes)
            for ax in range(3):
                if random.random() < 0.5:
                    g = np.flip(g, axis=ax)

            # 3) random translation (roll)
            shifts = [random.randint(-max_shift, max_shift) for _ in range(3)]
            g = np.roll(g, shift=shifts, axis=(0,1,2))

            # 4) random crop-and-pad
            crop_frac = random.uniform(0, max_crop_frac)
            if crop_frac > 0:
                crop_n = int(n * (1 - crop_frac))
                # ensure at least 1 voxel
                crop_n = max(crop_n, 1)
                start = [random.randint(0, n - crop_n) for _ in range(3)]
                cropped = g[
                    start[0]:start[0]+crop_n,
                    start[1]:start[1]+crop_n,
                    start[2]:start[2]+crop_n
                ]
                # pad back to original size
                pad_before = start
                pad_after = [n - crop_n - pb for pb in pad_before]
                g = np.pad(
                    cropped,
                    pad_width=[(pad_before[i], pad_after[i]) for i in range(3)],
                    mode='constant',
                    constant_values=False
                )

            # 5) random voxel dropout
            if drop_rate > 0:
                mask = (np.random.rand(*g.shape) < drop_rate) & g
                g[mask] = False

            # 6) random cuboid occlusion
            if random.random() < occlude_rate:
                sz = int(occlude_size_frac * n)
                dx = random.randint(1, max(1, sz))
                dy = random.randint(1, max(1, sz))
                dz = random.randint(1, max(1, sz))
                x0 = random.randint(0, n - dx)
                y0 = random.randint(0, n - dy)
                z0 = random.randint(0, n - dz)
                g[x0:x0+dx, y0:y0+dy, z0:z0+dz] = False

            # ensure boolean grid
            augs.append(g.astype(bool))

        return augs

class Contrastive_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_relu_block1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1), 
            nn.BatchNorm3d(16), 
            nn.ReLU()
        )
        self.conv_relu_block2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1), 
            nn.BatchNorm3d(32), 
            nn.ReLU()
        )
        self.conv_relu_block3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm3d(64), 
            nn.ReLU()
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)   # → [B, 64,1,1,1]
        self.proj_head   = nn.Sequential(            # contrastive projection
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 256)
        )

    def forward(self, x):
        x = self.conv_relu_block1(x)
        x = self.conv_relu_block2(x)
        x = self.conv_relu_block3(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        z = self.proj_head(x)
        return z

def make_pos_pair_indexes(batch_size, view_num, device):
    N = batch_size * view_num
    idx = torch.arange(N, device=device)
    obj_id = idx // view_num                               # which object each view belongs to
    # Compute all pairs (i,j) where obj_id[i] == obj_id[j] but i != j
    eq = obj_id.unsqueeze(1) == obj_id.unsqueeze(0)        # [N, N] mask
    neq = ~torch.eye(N, dtype=torch.bool, device=device)   # remove diagonal
    rows, cols = torch.where(eq & neq)
    return torch.stack([rows, cols], dim=1)                # [num_pairs, 2]

def train(dataloader, model, loss_fn, optimizer, sched, device, pos_pair_indexes):
    model.train()
    logger = logging.getLogger('train')
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch, k_tuplets in enumerate(dataloader):
        # k_tuplets: [B, V, D, H, W], float in [0,1]
        X = k_tuplets.to(device).float()
        B, V, D, H, W = X.shape

        # Flatten views into individual samples for the encoder
        X = X.unsqueeze(2)  # [B, V, 1, D, H, W]
        X = X.view(B * V, 1, D, H, W) 

        # Forward pass
        encodings = model(X)  # [N, C] where N = B*V

        # Compute loss
        loss = loss_fn(encodings, pos_pair_indexes.to(device))
        total_loss += loss.item()

        # Cosine similarity matrix [N, N]:
        S = F.cosine_similarity(
            encodings.unsqueeze(0),  # [1, N, C]
            encodings.unsqueeze(1),  # [N, 1, C]
            dim=-1
        )

        """
        # Uncoment this section if you want to monitor average positive pair distance and average negative pair distance in console
        
        # Zero out diagonal entries
        eye = torch.eye(S.size(0), device=S.device, dtype=torch.bool)
        S = S.masked_fill(eye, 0.0)
        
        # Gather positive similarities
        pi, pj = pos_pair_indexes[:,0], pos_pair_indexes[:,1]
        pos_vals = S[pi, pj]

        # Mask out positives and diagonal to get negatives
        neg_mask = torch.ones_like(S, dtype=torch.bool)
        neg_mask[eye] = False
        neg_mask[pi, pj] = False
        neg_vals = S[neg_mask].view(S.size(0), -1)

        avg_pos_sim = pos_vals.mean().item()
        avg_neg_sim = neg_vals.mean().item()

        print(f"Batch {batch:4d}: loss={loss.item():.4f}  "
        f"avg_pos_sim={avg_pos_sim:.4f}  avg_neg_sim={avg_neg_sim:.4f}")
        """
        # Log to both file logger and console
        logger.debug(f'Batch {batch}/{num_batches} Loss: {loss.item()}')

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Periodic checkpoint (TURN OFF IN PROD) 
        if (batch + 1) % 10 == 0:
            torch.save({
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': sched.state_dict(),
            }, "checkpoints/CNN_best.pth")

    # Step the scheduler once per epoch
    sched.step()
    avg_loss = total_loss / num_batches
    return avg_loss

def info_nce_loss(
    embeddings: torch.Tensor,
    pos_pair_indexes: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE (NT-Xent) loss for contrastive learning.

    Args:
        embeddings: Tensor of shape [N, D], raw encoder outputs (not yet normalized).
        pos_pair_indexes: LongTensor of shape [M, 2], each row (i, j) is a positive index pair.
            Assumes that for every i in [0..N), there is at least one (i, j) in pos_pair_indexes.
        temperature: Scaling factor for the cosine similarities.

    Returns:
        Scalar loss: cross-entropy treating one positive per anchor.
    """
    # 1) Normalize embeddings
    z = F.normalize(embeddings, dim=1)  # [N, D]

    # 2) Compute similarity logits
    logits = torch.matmul(z, z.t()) / temperature  # [N, N]

    # 3) Mask out self-similarities by setting diagonal to large negative
    N = logits.size(0)
    diag_mask = torch.eye(N, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diag_mask, -1e9)

    # 4) Build target indices: one positive per anchor
    #    We pick the first positive j for each anchor i.
    targets = torch.full((N,), -1, dtype=torch.long, device=logits.device)
    for i, j in pos_pair_indexes.tolist():
        if targets[i] < 0:
            targets[i] = j
    if (targets < 0).any():
        raise ValueError("Some embeddings have no positive pair!")

    # 5) Cross‐entropy loss
    loss = F.cross_entropy(logits, targets, reduction='mean')
    return loss

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])

def save_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = {
        'model_state': model.state_dict(),
    }
    if optimizer:
        ckpt['optimizer_state'] = optimizer.state_dict()
    if scheduler:
        ckpt['scheduler_state'] = scheduler.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch
    train_directory = args.data


    model = Contrastive_Encoder().to(device)
    # loss_fn = nt_bxent_loss
    loss_fn = info_nce_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Load or from scratch
    if os.path.isfile(args.ckpt):
        print(f"Resuming training from checkpoint: {args.ckpt}")
        load_checkpoint(args.ckpt, model, optimizer, sched, device)
    else:
        print("No checkpoint found, training from scratch.")

    train_files = []
    for root, _, fnames in os.walk(train_directory):
        if "test" in root.lower():
            continue
        for f in fnames:
            if f.lower().endswith(('.stl', '.obj', '.off')):
                train_files.append(os.path.join(root, f))
    if not train_files:
        print(f"No meshes found under {train_directory}")


    # Create data loaders.
    train_dataset = Voxelised_3D_Data(train_files)

    train_dataloader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=2,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=True,
                        prefetch_factor=2)
                

    # set up logging
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)

    # file handler for debug/info
    fh = logging.FileHandler('Voxel_training_batches.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    ))
    logger.addHandler(fh)

    # Grab one batch to infer B and V, then precompute pos‐pairs once:
    sample = next(iter(train_dataloader))
    B, V, D, H, W = sample.shape
    pos_pair_indexes = make_pos_pair_indexes(B, V, device)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):

        avg_loss = train(
            train_dataloader, model, loss_fn,
            optimizer, sched, device,
            pos_pair_indexes
        )
        print(f"Epoch {epoch:3d} | train loss: {avg_loss:.4f}")

        # only save if we improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(args.ckpt) or '.', exist_ok=True)
            torch.save({
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': sched.state_dict(),
                'epoch':           epoch,
                'best_loss':       best_loss,
            }, args.ckpt)
            print(f" → New best ({best_loss:.4f}), checkpoint saved.")
    print("Done!")

def index(paths, ckpt="checkpoints/CNN_best.pth", device='cpu'):
    embeddings = []
    dimensions = 64  # must match the Dataset's default
    enc = Contrastive_Encoder().to(device)
    load_checkpoint(ckpt, enc)
    enc.eval()

    for file in tqdm(paths):
        cache_path = file + ".voxel.pt"

        # 1) Try to load a properly-shaped cache
        voxel_grid = None
        if os.path.exists(cache_path):
            vg = torch.load(cache_path, map_location=device)
            if isinstance(vg, torch.Tensor) and vg.shape == (dimensions, dimensions, dimensions):
                voxel_grid = vg

        # 2) If missing or wrong shape, voxelize & resample
        # Optionally turn off caching of voxelisation here
        if voxel_grid is None:
            mesh = trimesh.load_mesh(file)
            # same cubic‐grid logic as __getitem__
            pitch = mesh.extents.max() / dimensions
            vg   = mesh.voxelized(pitch)
            vg_n = vg.revoxelized((dimensions, dimensions, dimensions))
            mat  = vg_n.matrix              # bool ndarray shape (D,D,D)
            voxel_grid = torch.from_numpy(mat)
            torch.save(voxel_grid, cache_path)

        # 3) Ensure tensor, float, and shape [1,1,D,D,D]
        if not isinstance(voxel_grid, torch.Tensor):
            voxel_grid = torch.from_numpy(np.array(voxel_grid))
        x = voxel_grid.float().unsqueeze(0).unsqueeze(0).to(device)

        # 4) Encode and normalize
        with torch.no_grad():
            z = enc(x)                           # [1, 256]
            z = F.normalize(z, dim=1)           # unit‐norm
            embeddings.append(z.squeeze(0).cpu().numpy())

    return embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ckpt', default='checkpoints/CNN_best.pth')
    args = parser.parse_args()
    run(args)