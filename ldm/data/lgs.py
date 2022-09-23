import os
import numpy as np
import pandas as pd
import PIL
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class LGSModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 data_root,
                 num_workers=None,
                 size=None,
                 repeats=100,
                 placeholder_token="*",
                 flip_p=0.5,
                 interpolation="bicubic",
                 center_crop=False):
        super().__init__()

        self.data_root = data_root
        self.size = size
        self.repeats = repeats
        self.placeholder_token = placeholder_token
        self.flip_p = flip_p
        self.interpolation = interpolation
        self.center_crop = center_crop

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2

    def prepare_data(self):
        metadata = pd.read_csv(f'{self.data_root}/list.csv')
        image_paths = [os.path.join(self.data_root, f_path)
                       for f_path in metadata['image'].values]
        captions = [caption for caption in metadata['caption'].values]
        self.data_full = list(zip(image_paths, captions))

    def setup(self, stage=None):
        train_set_size = int(len(self.data_full) * 0.8)
        valid_set_size = len(self.data_full) - train_set_size
        self.data_train, self.data_val = random_split(
            self.data_full, [train_set_size, valid_set_size])
        self.datasets = {'train': self.data_train, 'val': self.data_val}

    def train_dataloader(self):
        dataset = LGS(self.data_train, size=self.size, repeats=self.repeats, interpolation=self.interpolation,
                      flip_p=self.flip_p, placeholder_token=self.placeholder_token, center_crop=self.center_crop)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        dataset = LGS(self.data_val, size=self.size, interpolation=self.interpolation,
                      flip_p=self.flip_p, placeholder_token=self.placeholder_token, center_crop=self.center_crop)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class LGS(Dataset):
    def __init__(self,
                 data,
                 size=None,
                 repeats=1,
                 interpolation="bicubic",
                 flip_p=0.5,
                 placeholder_token="*",
                 center_crop=False,
                 ):

        self.data = data

        self.num_images = len(self.data)
        self._length = self.num_images * repeats

        self.placeholder_token = placeholder_token

        self.size = size
        self.center_crop = center_crop
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        image_path, text = self.data[i % self.num_images]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token

        example["caption"] = text.format(placeholder_string)

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
