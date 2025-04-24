import os
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from pytorch_lightning import seed_everything
from torchvision.io import read_image
from torchvision.utils import save_image
import torchvision

from masactrl.diffuser_utils_inversion_kv import MasaCtrlPipeline
from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.masactrl_inversion import MutualSelfAttentionControlInversion
from masactrl.masactrl_utils import AttentionBase, AttentionStore
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

torch.cuda.set_device(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_image(image_path, res, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.
    image = F.interpolate(image, (res, res))
    return image.to(device)

def load_mask(mask_path, res, device):
    mask = read_image(mask_path).float() / 255.
    if mask.shape[0] > 1:
        mask = mask.mean(dim=0, keepdim=True)  # convert to grayscale
    mask = F.interpolate(mask.unsqueeze(0), (res, res))
    return mask.to(device)

# Configuration
P1, P2 = 20, 5
S1, S2 = 20, 5
ref_images = torch.cat([
    load_image('./images/aug/203.jpg', 512, device),
    load_image('./images/aug/203-1.jpg', 512, device),
    load_image('./images/aug/203-2.jpg', 512, device),
    load_image('./images/aug/203-3.jpg', 512, device)
])
target_image = load_image('./images/tgts/203-1.jpg', 512, device)
target_mask = load_mask('./images/masks/203-1-mask.png', 512, device)

out_dir = "./workdir/exp/"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"Sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)

ref_num = ref_images.shape[0]
save_image(ref_images, os.path.join(out_dir, f"refs.jpg"), normalize=True)
save_image(target_image, os.path.join(out_dir, f"target.jpg"), normalize=True)

# Model
model_path = "CompVis/stable-diffusion-v1-4"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

# Invert references
start_code_ref, latents_list_ref = model.invert(ref_images, [""] * ref_num, num_inference_steps=50, return_intermediates=True)

# Invert target to get IR
_, latents_list_target_self = model.invert(target_image, "", num_inference_steps=50, return_intermediates=True)

# Self-rectification step 1 (visualization only)
editor = MutualSelfAttentionControlInversion(P1, 10, ref_num=1, masks=[target_mask]*ref_num)
regiter_attention_editor_diffusers(model, editor)
start_code_tgt, _ = model.invert(target_image, "", num_inference_steps=50, return_intermediates=True, ref_intermediate_latents=latents_list_target_self)

start_code = torch.cat([start_code_ref, start_code_tgt])

# Use AttentionStore to visualize step 1 attention
attention_store = AttentionStore()
regiter_attention_editor_diffusers(model, attention_store)
image_masactrl = model([""] * (ref_num) + [""], latents=start_code, ref_intermediate_latents=latents_list_ref)
save_image(image_masactrl[-1:], os.path.join(out_dir, f"result_01.jpg"))

# Save attention maps
attn_dir = os.path.join(out_dir, "attention_maps")
os.makedirs(attn_dir, exist_ok=True)
for i, attn in enumerate(attention_store.self_attns):
    avg_attn = attn[0].mean(dim=0)
    center_token_attn = avg_attn.mean(dim=0).reshape(64, 64)
    norm = (center_token_attn - center_token_attn.min()) / (center_token_attn.max() - center_token_attn.min() + 1e-8)
    torchvision.utils.save_image(norm.unsqueeze(0), os.path.join(attn_dir, f"attn_layer_{i}.png"))

# Self-rectification step 2 (region-masked)
TARGET_PATH = os.path.join(out_dir, f"result_01.jpg")
target_image = load_image(TARGET_PATH, 512, device)
editor = AttentionBase()
regiter_attention_editor_diffusers(model, editor)

editor = MutualSelfAttentionControlInversion(P2, 10, ref_num=1, masks=[target_mask]*ref_num)
regiter_attention_editor_diffusers(model, editor)
start_code_tgt, latents_list_tgt = model.invert(target_image, "", num_inference_steps=50, return_intermediates=True, ref_intermediate_latents=latents_list_target_self)

start_code = torch.cat([start_code_ref, start_code_tgt])
editor = MutualSelfAttentionControl(S2, 10, ref_num=ref_num, masks=[target_mask]*ref_num)
regiter_attention_editor_diffusers(model, editor)
image_masactrl = model([""] * (ref_num) + [""], latents=start_code, ref_intermediate_latents=latents_list_ref)
save_image(image_masactrl[-1:], os.path.join(out_dir, f"result_02.jpg"))

print("Synthesized images saved in", out_dir)
