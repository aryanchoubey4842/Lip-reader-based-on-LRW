# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LRWDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.samples = []

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            split_dir = os.path.join(root_dir, cls, split)
            if not os.path.isdir(split_dir):
                continue
            for fname in os.listdir(split_dir):
                if fname.endswith('.npy'):
                    self.samples.append((
                        os.path.join(split_dir, fname),
                        self.class_to_idx[cls]
                    ))

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(88, padding=8),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(88),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        video = np.load(path).astype(np.float32) / 255.0
        video = (video - 0.4161) / 0.1688

        video = torch.tensor(video).unsqueeze(1)  # (T, 1, H, W)
        frames = []
        for frame in video:
            frames.append(self.transform(frame))
        video = torch.stack(frames)

        return video, labelh