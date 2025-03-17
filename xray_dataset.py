from torch.utils.data import Dataset
from PIL import Image


class XRayDataset(Dataset):
    def __init__(self, image_paths_and_labels, transform=None):
        self.image_paths = [x[0] for x in image_paths_and_labels]
        self.labels = [x[1] for x in image_paths_and_labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
