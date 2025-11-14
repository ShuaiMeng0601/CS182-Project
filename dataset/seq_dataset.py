class SequenceDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        # expect a["actions"] shaped [N, 60, action_dim]
        self.actions = data["actions"]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # return shape [60, action_dim]
        a = torch.tensor(self.actions[idx], dtype=torch.float32)
        return a
