import os

from torchvision.io import read_image
import torch.nn.functional as F

class Utils:
    # ---------------------------------------------------
    @staticmethod
    def load_image(image_path:str, res, device):
        image = read_image(image_path)
        image = image[:3].unsqueeze_(0).float() / 127.5 - 1.
        image = F.interpolate(image, (res, res))
        return image.to(device)

    # ---------------------------------------------------
    @staticmethod
    def load_mask(mask_path:str, res, device):
        if not os.path.exists(mask_path):
            return None
        mask = read_image(mask_path).float() / 255.
        if mask.shape[0] > 1:
            mask = mask.mean(dim=0, keepdim=True)
        mask = F.interpolate(mask.unsqueeze(0), (res, res))
        return mask.to(device)