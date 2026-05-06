import torch
from torch import nn


ESM_MODEL_NAME = "esm2_t30_150M_UR50D"
ESM_LAYER = 30
ESM_EMBED_DIM = 640
LABEL_TO_ID = {"H": 0, "E": 1, "C": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class ResidueMLPClassifier(nn.Module):
    """Per-residue MLP classifier for frozen ESM2 embeddings."""

    def __init__(self, input_dim=ESM_EMBED_DIM, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)


def count_learned_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

