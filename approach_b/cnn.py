import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import ChordDataset 
import matplotlib.pyplot as plt


class CNN1DClassifier(nn.Module):
    def __init__(self, max_len: int):
        """
        A 1D-CNN that maps (B,1,max_len) → (B,2)
        Architecture:
          Conv1d(1→16, k=9, p=4) + ReLU + BN + MaxPool(k=4)
          Conv1d(16→32, k=9, p=4) + ReLU + BN + MaxPool(k=4)
          Flatten
          FC(32 * (max_len/16) → 64) + ReLU
          FC(64 → 2)
        """
        super().__init__()

        # conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=4),
        )
        # conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=4),
        )

        # compute flattened feature size after two 4× pools
        pooled_len = max_len // 4 // 4  # == max_len // 16
        feat_dim = 32 * pooled_len

        # fully-connected head
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (B, 32, pooled_len) → (B, feat_dim)
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, max_len)
        returns logits of shape (B, 2)
        """
        x = self.conv1(x)    # → (B,16, max_len/4)
        x = self.conv2(x)    # → (B,32, max_len/16)
        x = self.classifier(x)
        return x

def main():
    # Hyper-params
    num_epochs = 20
    batch_size = 16
    lr = 1e-3

    # 1) Data + model + loss + optimizer
    dataset = ChordDataset("/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files", sr=44100)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = CNN1DClassifier(max_len=dataset.max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 2) Prepare history trackers
    loss_history = []
    acc_history  = []

    # 3) Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            running_corrects += (preds == y_batch).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc  = running_corrects / len(dataset)

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print(f"Epoch {epoch:3d}  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

    # 4) Plotting
    epochs = list(range(1, num_epochs+1))

    plt.figure()
    plt.plot(epochs, loss_history, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.figure()
    plt.plot(epochs, acc_history, marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()

