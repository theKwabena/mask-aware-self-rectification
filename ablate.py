# ---------------------------------------------------
# Stdlib import
import os
import torch
import argparse

# ---------------------------------------------------
# 3rd Party Imports
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image

from diffusers import DDIMScheduler

# ---------------------------------------------------
# Local Imports
from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.diffuser_utils_inversion_kv import MasaCtrlPipeline
from masactrl.masactrl_inversion import MutualSelfAttentionControlInversion
from masactrl.masactrl_utils import regiter_attention_editor_diffusers, AttentionBase
from utils.utils import Utils

# ---------------------------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# ---------------------------------------------------
class Pipeline:

    # ---------------------------------------------------
    @staticmethod
    def run(sample_id, use_mask):
        out_dir = os.path.join("ablation_results", sample_id, "global_mask" if use_mask else "no_mask")
        os.makedirs(out_dir, exist_ok=True)

        ref_image_path = f"assets/{sample_id}/ref.jpg"
        target_image_path = f"assets/{sample_id}/target.jpg"
        mask_path = f"assets/{sample_id}/mask.png"

        ref_image = Utils.load_image(ref_image_path, 512, device)
        target_image = Utils.load_image(target_image_path, 512, device)
        target_mask = Utils.load_mask(mask_path, 512, device) if use_mask else None

        ref_images = ref_image.repeat(4, 1, 1, 1)  # Simulate 4 augmented references
        ref_num = ref_images.shape[0]

        save_image(ref_images, os.path.join(out_dir, 'refs.jpg'), normalize=True)
        save_image(target_image, os.path.join(out_dir, 'target.jpg'), normalize=True)

        model_path = "CompVis/stable-diffusion-v1-4"
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

        P1, P2 = 20, 5
        S1, S2 = 20, 5

        start_code_ref, latents_list_ref = model.invert(ref_images, [""] * ref_num, num_inference_steps=50, return_intermediates=True)
        _, latents_list_target_self = model.invert(target_image, "", num_inference_steps=50, return_intermediates=True)

        editor = MutualSelfAttentionControlInversion(P1, 10, ref_num=1, mask=target_mask if use_mask else None)
        regiter_attention_editor_diffusers(model, editor)
        start_code_tgt, _ = model.invert(target_image, "", num_inference_steps=50, return_intermediates=True, ref_intermediate_latents=latents_list_target_self)
        start_code = torch.cat([start_code_ref, start_code_tgt])

        editor = MutualSelfAttentionControl(S1, 10, ref_num=ref_num, mask=target_mask if use_mask else None)
        regiter_attention_editor_diffusers(model, editor)
        image_masactrl = model([""] * (ref_num) + [""], latents=start_code, ref_intermediate_latents=latents_list_ref)
        save_image(image_masactrl[-1:], os.path.join(out_dir, 'result_01.jpg'))

        editor = AttentionBase()
        regiter_attention_editor_diffusers(model, editor)
        target_image = Utils.load_image(os.path.join(out_dir, 'result_01.jpg'), 512, device)

        editor = MutualSelfAttentionControlInversion(P2, 10, ref_num=1, mask=target_mask if use_mask else None)
        regiter_attention_editor_diffusers(model, editor)
        start_code_tgt, _ = model.invert(target_image, "", num_inference_steps=50, return_intermediates=True, ref_intermediate_latents=latents_list_target_self)
        start_code = torch.cat([start_code_ref, start_code_tgt])

        editor = MutualSelfAttentionControl(S2, 10, ref_num=ref_num, mask=target_mask if use_mask else None)
        regiter_attention_editor_diffusers(model, editor)
        image_masactrl = model([""] * (ref_num) + [""], latents=start_code, ref_intermediate_latents=latents_list_ref)
        save_image(image_masactrl[-1:], os.path.join(out_dir, 'result_02.jpg'))

        print(f"[{sample_id} - {'global_mask' if use_mask else 'no_mask'}] Done -> {out_dir}")

# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Sample ID (e.g. 203, 3001)")
    args = parser.parse_args()

    # ---------------------------------------------------
    Pipeline.run(args.id, use_mask=False)
    Pipeline.run(args.id, use_mask=True)
