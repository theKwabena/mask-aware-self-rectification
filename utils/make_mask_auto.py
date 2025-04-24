import os
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.models.segmentation import deeplabv3_resnet50
import argparse
import cv2


def generate_mask(image_path, out_path, target_class=15):
    img = Image.open(image_path).convert("RGB")
    transform = Compose([
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)

    model = deeplabv3_resnet50(pretrained=True).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        seg = output.argmax(0).byte().cpu().numpy()

    mask = (seg == target_class).astype(np.uint8) * 255
    cv2.imwrite(out_path, mask)
    print(f"Saved mask to {out_path}")


def auto_generate_masks(assets_dir):
    for subdir in sorted(os.listdir(assets_dir)):
        path = os.path.join(assets_dir, subdir)
        if os.path.isdir(path):
            target_img = os.path.join(path, "target.jpg")
            out_mask = os.path.join(path, "mask.png")
            if os.path.exists(target_img):
                generate_mask(target_img, out_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="assets", help="Path to assets folder")
    args = parser.parse_args()

    auto_generate_masks(args.assets)