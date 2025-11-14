# dataset/seq_dataset.py
import torch
from torch.utils.data import Dataset
import pickle

class SequenceDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        # each item must be np.array or tensor shaped [seq_len, dim]
        self.states = data["states"]            # shape [N, state_dim]
        self.actions = data["actions"]          # shape [N, 60, action_dim]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        s = torch.tensor(self.states[idx], dtype=torch.float32)
        a = torch.tensor(self.actions[idx], dtype=torch.float32)
        x = torch.cat([s.flatten(), a.flatten()], dim=-1)
        return x
