import torch
from einops import rearrange

from .masactrl_utils import AttentionBase

# ---------------------------------------------------
class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    # ---------------------------------------------------
    def __init__(self, start_step=4, start_layer=10, ref_num=1, mask=None, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        self.ref_num = ref_num
        self.mask = mask
        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at U-Net layers: ", self.layer_idx)
        print("ref_num = ", self.ref_num)

    # ---------------------------------------------------
    def apply_mask(self, tensor, num_heads):
        if self.mask is None:
            return tensor
        B, N, D = tensor.shape
        H = W = int(N ** 0.5)
        resized_mask = F.interpolate(self.mask, size=(H, W), mode='nearest')
        flat_mask = resized_mask.view(1, 1, -1).expand(B, -1, -1).reshape(B * N).unsqueeze(-1)
        return tensor * flat_mask

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    # ---------------------------------------------------
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        k = self.apply_mask(k, num_heads)
        v = self.apply_mask(v, num_heads)

        out_c_ref = super().forward(
            q[:num_heads * self.ref_num],
            k[:num_heads * self.ref_num],
            v[:num_heads * self.ref_num],
            sim[:num_heads * self.ref_num],
            attn[:num_heads * self.ref_num],
            is_cross, place_in_unet, num_heads, **kwargs)

        out_c_tgt = self.attn_batch(
            q[num_heads * self.ref_num:],
            k[:num_heads * self.ref_num],
            v[:num_heads * self.ref_num],
            sim[:num_heads * self.ref_num],
            attn, is_cross, place_in_unet, num_heads, **kwargs)

        out = torch.cat([out_c_ref, out_c_tgt], dim=0)
        return out

