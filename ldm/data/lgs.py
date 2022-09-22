import os
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LGS(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 center_crop=False,
                 per_image_tokens=False
                 ):

        self.data_root = data_root
        self.metadata = pd.read_csv(f'{self.data_root}/list.csv')

        self.image_paths = [os.path.join(self.data_root, f_path) for f_path in self.metadata['image'].values]
        self.captions = [caption for caption in self.metadata['caption'].values]

        train_set_size = int(len(self.image_paths) * 0.8)

        if set == "train":
            self.image_paths = self.image_paths[:train_set_size]
            self.captions = self.captions[:train_set_size]
        else:
            self.image_paths = self.image_paths[train_set_size:]
            self.captions = self.captions[train_set_size:]

        self.num_images = len(self.image_paths)
        self._length = self.num_images
        
        self.placeholder_token = placeholder_token

        if set == "train":
            self._length = self.num_images * repeats

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

        text = self.captions[i % self.num_images]
        image = Image.open(self.image_paths[i % self.num_images])

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
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
