import torch
from typing import Tuple, Union, Any
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


class MnistSet(Dataset):
    def __init__(self, split_ratio=80, pad_value=-1, min_pixel_brightness=0):
        self.image_size = 28
        self.element_size = 2
        self.max_num_elements = 360
        self.pad_value = pad_value
        self.min_pixel_brightness = min_pixel_brightness

        # Transform: convert to tensor and normalize 0–1
        transform = transforms.Compose([transforms.ToTensor()])

        # Load full MNIST dataset (train + test merged)
        full_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        self.full_dataset = torch.utils.data.ConcatDataset([full_data, test_data])

        total_len = len(self.full_dataset)
        train_len = int(total_len * (split_ratio / 100.0))
        val_len = total_len - train_len
        splits = random_split(self.full_dataset, [train_len, val_len])
        self.train_set, self.val_set = splits

        self.active_split = "train"

    def set_split(self, split="train"):
        assert split in ["train", "val"], "Split must be 'train' or 'val'"
        self.active_split = split

    def __len__(self):
        if self.active_split == "train":
            return len(self.train_set)
        else:
            return len(self.val_set)

    def __getitem__(self, idx):
        if self.active_split == "train":
            subset: Union[Subset, Dataset] = self.train_set
        else:
            subset = self.val_set

        if hasattr(subset, "indices") and hasattr(subset, "dataset"):
            orig_idx = subset.indices[idx]  
            img, label = subset.dataset[orig_idx]  
        else:
            
            from typing import cast, Tuple

            sample = subset[idx]
            sample_t = cast(Tuple[Any, Any], sample)
            try:
                img, label = sample_t[0], sample_t[1]
            except Exception:
                raise TypeError("Unable to extract (img, label) from subset sample")

        img_np = img.squeeze().numpy()
        coords = np.argwhere(img_np > self.min_pixel_brightness / 255.0)

        coords = coords / self.image_size

        size = coords.shape[0]
        padded = np.full((self.max_num_elements, 2), self.pad_value, dtype=np.float32)
        if size > 0:
            n = min(size, self.max_num_elements)
            padded[:n, :] = coords[:n, :]

        return torch.tensor(img_np, dtype=torch.float32), torch.tensor(padded, dtype=torch.float32), torch.tensor(size), torch.tensor(label)

    def get_train_loader(self, batch_size=32, shuffle=True):
        self.set_split("train")
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    def get_val_loader(self, batch_size=32, shuffle=False):
        self.set_split("val")
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=False)


if __name__ == "__main__":
    dataset = MnistSet(split_ratio=80, pad_value=-999, min_pixel_brightness=50)
    loader = dataset.get_train_loader(batch_size=1)

    for raw, pixels, size, label in loader:
        raw = raw.squeeze(0).numpy()
        pixels = pixels.squeeze(0).numpy()

        x, y = pixels[:, 1], pixels[:, 0]
        valid = x != -999  # filter padding

        plt.imshow(raw, cmap="gray")
        plt.scatter(x[valid], y[valid], s=3, c="red")
        plt.axis((0.0, 1.0, 0.0, 1.0))
        plt.title(f"Label: {label.item()}, Set Size: {size.item()}")
        plt.show()
