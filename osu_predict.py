import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

class MultiLabelClassifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.fc1 = nn.Linear(2,64)
        self.fc2 = nn.Linear(64,128)
        self.out = nn.Linear(128,n_labels)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

id_list = np.load("id_list.npy", allow_pickle=True)
model = MultiLabelClassifier(n_labels=len(id_list))
model.load_state_dict(torch.load("osu_nn.pth"))

def predict_ids(model, rating, archetype, k=12):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[rating, archetype]], dtype=torch.float32)
        probs = model(x).squeeze()
        topk_indices = torch.topk(probs, k=k).indices
        return [id_list[i] for i in topk_indices]

result = predict_ids(model, rating=22000, archetype=100)

print(result)
