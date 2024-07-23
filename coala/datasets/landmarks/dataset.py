import os
import csv
import torch.utils.data as data
from PIL import Image


def read_csv(path: str):
    """Reads a csv file, and returns the content inside a list of dictionaries.
    Args:
      path: The path to the csv file.
    Returns:
      A list of dictionaries. Each row in the csv file will be a list entry. The
      dictionary is keyed by the column names.
    """
    with open(path, "r") as f:
        return list(csv.DictReader(f))


class LandmarksDataset(data.Dataset):
    def __init__(
            self,
            data_dir,
            local_files,
            transform=None,
            target_transform=None,
    ):
        """
        allfiles is [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ...
                     {'user_id': xxx, 'image_id': xxx, 'class': xxx} ... ]
        """
        self.local_files = local_files
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.local_files)

    def __getitem__(self, idx):
        img_name = self.local_files[idx]["image_id"]
        label = int(self.local_files[idx]["class"])

        img_name = os.path.join(self.data_dir, str(img_name) + ".jpg")
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
