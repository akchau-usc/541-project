import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ChordDataset(Dataset):
    """
    A Dataset of .wav files labeled as major (1) or minor (0) chords.
    1) Scans all files to compute the maximum length in samples.
    2) On the fly: loads, normalizes, pads/truncates to max_len, and returns a (1, max_len) tensor.
    """

    def __init__(self,
                 root_dir: str,
                 sr: int = 44100,
                 epsilon: float = 1e-8):
        """
        Args:
            root_dir: path containing two subfolders 'major' and 'minor', each with .wav files.
            sr:      target sampling rate for librosa.load
            epsilon: small constant to avoid divide-by-zero in normalization
        """
        self.sr = sr
        self.epsilon = epsilon

        # gather all file paths and integer labels
        self.file_paths = []
        self.labels = []
        for chord_type, label in (("minor", 0), ("major", 1)):
            folder = os.path.join(root_dir, chord_type)
            for fname in os.listdir(folder):
                if fname.lower().endswith(".wav"):
                    self.file_paths.append(os.path.join(folder, fname))
                    self.labels.append(label)

        # pre-compute maximum length in samples
        self.max_len = self._compute_max_len()

    def _compute_max_len(self) -> int:
        max_len = 0
        for path in self.file_paths:
            # use librosa.get_duration (avoids loading whole file twice)
            dur = librosa.get_duration(filename=path)
            length = int(dur * self.sr)
            if length > max_len:
                max_len = length
        return max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1) load + resample
        path = self.file_paths[idx]
        x, _ = librosa.load(path, sr=self.sr)  # x is shape (n_samples,)

        # 2) normalize
        mu = np.mean(x)
        sigma = np.std(x)
        x = (x - mu) / (sigma + self.epsilon)

        # 3) pad or truncate to max_len
        if len(x) < self.max_len:
            pad_width = self.max_len - len(x)
            x = np.pad(x, (0, pad_width), mode="constant")
        else:
            x = x[: self.max_len]

        # to tensor, shape (1, max_len)
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x_tensor, y

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    DATA_DIR = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files"   # should contain /minor/*.wav and /major/*.wav

    dataset = ChordDataset(DATA_DIR, sr=44100)
    print(f"Found {len(dataset)} examples; max_len = {dataset.max_len} samples")

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # iterate one batch
    for batch_x, batch_y in loader:
        # batch_x: Tensor of shape (B, 1, max_len)
        # batch_y: Tensor of shape (B,)
        print(batch_x.shape, batch_y)
        break
