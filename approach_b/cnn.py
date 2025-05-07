import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report

class MelSpecChordDataset(Dataset):
    """
    A Dataset of .wav files labeled as major (1) or minor (0) chords.
    Extracts for each clip:
      • log-Mel spectrogram → shape (n_mels, max_frames)
    Pads/truncates to a fixed size so every example is (1, n_mels, max_frames).
    """
    def __init__(self,
                 root_dir: str,
                 sr: int = 44100,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 64,
                 epsilon: float = 1e-8):
        self.sr          = sr
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_mels      = n_mels
        self.epsilon     = epsilon

        self.file_paths = []
        self.labels     = []
        for chord_type, label in (("minor", 0), ("major", 1)):
            folder = os.path.join(root_dir, chord_type)
            for fname in os.listdir(folder):
                if fname.lower().endswith(".wav"):
                    self.file_paths.append(os.path.join(folder, fname))
                    self.labels.append(label)

        # find longest clip in samples → then max_frames
        max_samples = 0
        for path in self.file_paths:
            dur = librosa.get_duration(filename=path)
            samples = int(dur * self.sr)
            if samples > max_samples:
                max_samples = samples
        self.max_samples = max_samples
        self.max_frames  = int(np.ceil(self.max_samples / self.hop_length))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        y, _ = librosa.load(path, sr=self.sr)

        # pad/truncate to max_samples
        if len(y) < self.max_samples:
            y = np.pad(y, (0, self.max_samples - len(y)), mode="constant")
        else:
            y = y[:self.max_samples]

        # normalize to zero mean/unit var
        mu, std = np.mean(y), np.std(y)
        y = (y - mu) / (std + self.epsilon)

        # compute log-Mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )  # (n_mels, t)
        S_db = librosa.power_to_db(S, ref=np.max)

        # pad/truncate along time axis to max_frames
        t = S_db.shape[1]
        if t < self.max_frames:
            S_db = np.pad(S_db, ((0,0),(0,self.max_frames - t)), mode="constant")
        else:
            S_db = S_db[:, : self.max_frames]

        # to tensor shape (1, n_mels, max_frames)
        x = torch.from_numpy(S_db).float().unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

class CNN2DClassifier(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int):
        """
        A small 2D‐CNN for (in_channels, height, width) inputs.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        # infer flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            out = self.conv1(dummy)
            out = self.conv2(out)
            out = self.conv3(out)
            feat_dim = out.shape[1] * out.shape[2] * out.shape[3]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

def main():
    # ─── Hyper-parameters ─────────────────────────
    num_epochs   = 20
    val_split    = 0.2
    batch_size   = 16
    lr           = 0.1
    dataset_path = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files"

    # ─── Prepare dataset & loaders ─────────────────
    full_dataset = MelSpecChordDataset(dataset_path,
                                       sr=44100,
                                       n_fft=2048,
                                       hop_length=512,
                                       n_mels=64)
    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # ─── Model, loss, optimizer ────────────────────
    model  = CNN2DClassifier(
        in_channels=1,
        height=full_dataset.n_mels,
        width=full_dataset.max_frames
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ─── History trackers ──────────────────────────
    train_loss_hist, train_acc_hist = [], []
    val_loss_hist,   val_acc_hist   = [], []

    # ─── Training & Validation loops ──────────────
    device = next(model.parameters()).device
    for epoch in range(1, num_epochs+1):
        # training
        model.train()
        running_loss = running_corrects = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss     += loss.item() * x_batch.size(0)
            running_corrects += (logits.argmax(1) == y_batch).sum().item()

        train_loss = running_loss / train_size
        train_acc  = running_corrects / train_size
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # validation
        model.eval()
        val_running_loss = val_running_corrects = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss   = criterion(logits, y_batch)
                val_running_loss     += loss.item() * x_batch.size(0)
                val_running_corrects += (logits.argmax(1) == y_batch).sum().item()

        val_loss = val_running_loss / val_size
        val_acc  = val_running_corrects / val_size
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        print(
            f"Epoch {epoch:2d} | "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
            f"val_loss:   {val_loss:.4f}   val_acc:   {val_acc:.4f}"
        )

    # ─── Confusion matrix & F1 on validation set ───
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            logits  = model(x_batch)
            preds   = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_batch.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print("\nConfusion Matrix:\n", cm)
    print(f"Macro F1 Score: {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["minor","major"]))

    # ─── Plotting ──────────────────────────────────
    epochs = range(1, num_epochs+1)

    plt.figure()
    plt.plot(epochs, train_loss_hist, marker='o', label="Train Loss")
    plt.plot(epochs,   val_loss_hist, marker='x', label="Val Loss")
    plt.title(f"Loss vs. Epoch (lr={lr})")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(epochs, train_acc_hist, marker='o', label="Train Acc")
    plt.plot(epochs,   val_acc_hist, marker='x', label="Val Acc")
    plt.title(f"Accuracy vs. Epoch (lr={lr})")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
