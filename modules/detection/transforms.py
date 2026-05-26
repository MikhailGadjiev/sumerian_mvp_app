"""PIL + box transforms for detection (minimal, torchvision-style)."""

from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: dict):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PILToTensor:
    def __call__(self, image: Image.Image, target: dict):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: Image.Image, target: dict):
        if random.random() >= self.prob:
            return image, target
        w, _ = image.size
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1
            target["boxes"] = boxes
        return image, target
