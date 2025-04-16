import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer

data = np.loadtxt("player_archetypes.csv", delimiter=",")

Xmat = data[:,:2]
Ymat = data[:,2:]

mlb = MultiLabelBinarizer()
id_labels = mlb.fit_transform(Ymat)

X_tensor = torch.tensor(Xmat, dtype=torch.float32)
Y_tensor = torch.tensor(id_labels, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

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

model = MultiLabelClassifier(n_labels = id_labels.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss() # good for multi label classification

for epoch in range(50):
    total_loss = 0
    for xb,yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)

torch.save(model.state_dict(), "player_archetype_nn.pth")