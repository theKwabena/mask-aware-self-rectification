from .masactrl import MutualSelfAttentionControl

class MutualSelfAttentionControlInversion(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, ref_num=1, mask=None, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        super().__init__(start_step, start_layer, ref_num, mask, layer_idx, step_idx, total_steps, model_type)
        self.step_idx = list(range(0, start_step))
        print("Inversion Control active during steps:", self.step_idx)



