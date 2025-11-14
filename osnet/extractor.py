import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .model import osnet_ain_x1_0, osnet_ibn_x1_0, osnet_x1_0
from .utils import load_weights_from_file


class OSNetFeatureExtractor(object):
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        model_path: str = "",
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device: str = "cuda",
    ):
        supported = {
            "osnet_x1_0": osnet_x1_0,
            "osnet_ibn_x1_0": osnet_ibn_x1_0,
            "osnet_ain_x1_0": osnet_ain_x1_0,
        }
        if model_name not in supported:
            raise ValueError(
                f"Unsupported model_name: {model_name}. Supported: {list(supported.keys())}"
            )

        model = supported[model_name](num_classes=1, loss="softmax")
        if model_path:
            load_weights_from_file(model, model_path)
        model.eval()

        transforms = [T.Resize(image_size), T.ToTensor()]
        if pixel_norm:
            transforms.append(T.Normalize(mean=pixel_mean, std=pixel_std))
        preprocess = T.Compose(transforms)

        self.to_pil = T.ToPILImage()
        self.preprocess = preprocess
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def __call__(self, input):
        if isinstance(input, list):
            images = []
            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert("RGB")
                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)
                else:
                    raise TypeError("Elements must be str or numpy.ndarray")
                image = self.preprocess(image)
                images.append(image)
            images = torch.stack(images, dim=0).to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert("RGB")
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            feats = self.model(images)
        return feats
