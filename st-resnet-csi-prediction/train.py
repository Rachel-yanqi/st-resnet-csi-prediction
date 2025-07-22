
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from generate_csi import generate_spatial_temporal_correlated_csi, exponential_corr_matrix
from st_resnet import STResNet

class STResNetCSIDataset(Dataset):
    def __init__(self, num_samples, num_users, history, augment_rhos, augment_decays):
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            rho = random.choice(augment_rhos)
            decay_tx = random.choice(augment_decays)
            decay_rx = random.choice(augment_decays)
            R_tx = exponential_corr_matrix(num_users, decay=decay_tx)
            R_rx = exponential_corr_matrix(num_users, decay=decay_rx)
            csi = generate_spatial_temporal_correlated_csi(num_symbols=history+1, num_users=num_users, rho=rho, R_tx=R_tx, R_rx=R_rx)
            csi_real = np.stack([np.real(csi), np.imag(csi)], axis=1)
            self.data.append(csi_real[:-1])
            self.labels.append(csi_real[-1])
        self.data = torch.tensor(np.stack(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.stack(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example training loop
model = STResNet()
dataset = STResNetCSIDataset(2000, 4, 5, [0.95, 0.98, 0.99], [0.6, 0.8, 0.9])
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
