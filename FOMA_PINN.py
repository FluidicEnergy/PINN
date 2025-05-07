import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob

# -----------------------------
# 1. Dataset for OpenFOAM Data
# -----------------------------
class OpenFOAMDataset(Dataset):
    def __init__(self, data_folder, N_prev=1, dt=0.5):
        """
        data_folder: path containing files:
            vel_0.npy, vel_0.5.npy, ..., each of shape (Nx, Ny, 2) for (u, v)
            pres_0.npy, pres_0.5.npy, ..., each of shape (Nx, Ny) for pressure
        N_prev: number of previous time steps to use
        dt: time interval between frames
        """
        self.N_prev = N_prev
        self.dt = dt

        # Load velocity and pressure snapshots sorted by time
        vel_files  = sorted(glob.glob(f"{data_folder}/vel_*.npy"))
        pres_files = sorted(glob.glob(f"{data_folder}/pres_*.npy"))
        self.vel_data  = np.stack([np.load(f) for f in vel_files], axis=0)  # (T, Nx, Ny, 2)
        self.pres_data = np.stack([np.load(f) for f in pres_files], axis=0) # (T, Nx, Ny)

        self.T, self.Nx, self.Ny, _ = self.vel_data.shape
        # Create flattened spatial coordinates
        xs = np.linspace(0, 1, self.Nx)
        ys = np.linspace(0, 1, self.Ny)
        coords = np.stack(np.meshgrid(xs, ys), axis=-1)  # (Ny, Nx, 2)
        self.coords = coords.reshape(-1, 2)               # (Nx*Ny, 2)

        # Prepare sample indices (time index, spatial index)
        self.samples = [
            (t, idx)
            for t in range(self.N_prev, self.T)
            for idx in range(self.coords.shape[0])
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        t, idx = self.samples[i]
        x, y = self.coords[idx]

        # Gather previous snapshots
        u_prev = []
        v_prev = []
        p_prev = []
        for n in range(self.N_prev):
            tp = t - n - 1
            u_prev.append(self.vel_data[tp, :, :, 0].reshape(-1)[idx])
            v_prev.append(self.vel_data[tp, :, :, 1].reshape(-1)[idx])
            p_prev.append(self.pres_data[tp].reshape(-1)[idx])
        uvp_prev = np.concatenate([u_prev, v_prev, p_prev])

        # Prediction time
        t_pred = t * self.dt

        # Targets at time t
        u_t = self.vel_data[t, :, :, 0].reshape(-1)[idx]
        v_t = self.vel_data[t, :, :, 1].reshape(-1)[idx]
        p_t = self.pres_data[t].reshape(-1)[idx]

        inputs = {
            'xy':        torch.tensor([x, y], dtype=torch.float32),
            't':         torch.tensor([t_pred], dtype=torch.float32),
            'uvp_prev':  torch.tensor(uvp_prev, dtype=torch.float32),
        }
        target = torch.tensor([u_t, v_t, p_t], dtype=torch.float32)
        return inputs, target

# -----------------------------
# 2. DeepONet Architecture
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, N_prev=1, embed_dim=64):
        super().__init__()
        # Three branch networks
        self.branch_xy   = MLP(in_dim=2,             hidden_dim=128, out_dim=embed_dim)
        self.branch_t    = MLP(in_dim=1,             hidden_dim=64,  out_dim=embed_dim)
        self.branch_uvp  = MLP(in_dim=3 * N_prev,    hidden_dim=128, out_dim=embed_dim)
        # Trunk network to fuse embeddings
        self.trunk = MLP(in_dim=embed_dim*3, hidden_dim=128, out_dim=3)  # predict (u, v, p)

    def forward(self, xy, t, uvp_prev):
        b1 = self.branch_xy(xy)        # (batch, embed_dim)
        b2 = self.branch_t(t)          # (batch, embed_dim)
        b3 = self.branch_uvp(uvp_prev) # (batch, embed_dim)
        features = torch.cat([b1, b2, b3], dim=1)
        out = self.trunk(features)     # (batch, 3)
        return out

# -----------------------------
# 3. Training Loop
# -----------------------------
def train(model, dataloader, epochs=50, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_target in dataloader:
            xy       = batch_inputs['xy'].to(device)
            t        = batch_inputs['t'].to(device)
            uvp_prev = batch_inputs['uvp_prev'].to(device)
            y_true   = batch_target.to(device)

            y_pred = model(xy, t, uvp_prev)
            loss = criterion(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_true.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:03d} - Loss: {avg_loss:.6f}")

# -----------------------------
# 4. Example Usage
# -----------------------------
if __name__ == "__main__":
    data_folder = "path/to/openfoam/data"
    dataset     = OpenFOAMDataset(data_folder, N_prev=1, dt=0.5)
    loader      = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=4)

    model = DeepONet(N_prev=1, embed_dim=64)
    train(model, loader, epochs=100, lr=1e-3, device='cuda')

