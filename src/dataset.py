import h5py
import torch
from torch.utils.data import DataLoader, Dataset


class H5Dataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            # X: (N, 1, 128, 128) as float32
            self.X = torch.from_numpy(f["X"][:]).float().unsqueeze(1)
            # Y: (N, 3) as float32
            self.Y = torch.from_numpy(f["Y"][:]).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == "__main__":
    # Example run:
    SIM_PATH = "data/project2_simulation_103k_norm16bin.h5"
    EXP_PATH = "data/project2_experiment_norm16bin.h5"

    train_loader = DataLoader(H5Dataset(SIM_PATH), batch_size=128, shuffle=True)
    val_loader = DataLoader(H5Dataset(EXP_PATH), batch_size=256, shuffle=False)

    x, y = next(iter(train_loader))
    print(x.shape, y.shape)  # (128, 1, 128, 128) (128, 3)
