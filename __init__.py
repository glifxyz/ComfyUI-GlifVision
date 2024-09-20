import math
import folder_paths
import os
import comfy.utils
import comfy.model_management
from .glif_vision import GlifVision
from .flux_hack import double_stream_forward, single_stream_forward
from functools import partial
from comfy.samplers import calculate_sigmas

if "glif_vision" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["glif_vision"] = (
        [os.path.join(folder_paths.models_dir, "glif_vision")], folder_paths.supported_pt_extensions
    )


class GlifVisionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glif_vision_file": (folder_paths.get_filename_list("glif_vision"),)
            }
        }

    RETURN_TYPES = ("GLIF_VISION",)
    FUNCTION = "load_glif_vision_model"
    CATEGORY = "glif_vision"

    def load_glif_vision_model(self, glif_vision_file):
        ckpt_path = folder_paths.get_full_path("glif_vision", glif_vision_file)
        state_dict = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        print("glif vision loaded")
        glif_vision = GlifVision(state_dict)
        glif_vision.eval()

        return (glif_vision,)


def add_block_forward_patch(model, patch_kwargs, key, forward_fn):
    block_type, block_id = key  # double_blocks, 0
    # bind the new forward
    target_module = getattr(model.model.diffusion_model, block_type)[block_id]
    new_forward = partial(forward_fn, target_module)

    # if glif_vision_patch_kwargs is not an attribute, create it
    if not hasattr(target_module, "glif_vision_patch_kwargs"):
        setattr(target_module, "glif_vision_patch_kwargs", None)

    attr_string = f"diffusion_model.{block_type}.{block_id}.forward"
    kwarg_string = f"diffusion_model.{block_type}.{block_id}.glif_vision_patch_kwargs"

    # target_module = model.model.diffusion_model.double_blocks[id]
    # do it based on block_type

    model.add_object_patch(attr_string, new_forward)
    model.add_object_patch(kwarg_string, {**patch_kwargs})
    return model


def hack_forward_to_save_kwargs(model):
    # currently flux pass along the transformer_options to the forward function
    # we need to save it on every block because we need it in the forward function
    attr_string = f"diffusion_model.forward"

    target_module = model.model.diffusion_model

    orig_forward = target_module.forward

    def forward_fn(self, x, timestep, context, y, guidance, control=None, **kwargs):
        transformer_options = kwargs.get("transformer_options", None)
        self.transformer_options = transformer_options
        for block in self.double_blocks:
            block.transformer_options = transformer_options
        for block in self.single_blocks:
            block.transformer_options = transformer_options

        return orig_forward(x, timestep, context, y, guidance, control, **kwargs)

    new_forward = partial(forward_fn, target_module)

    model.add_object_patch(attr_string, new_forward)


class GlifVisionApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glif_vision": ("GLIF_VISION",),
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "attn_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_glif_vision"
    CATEGORY = "glif_vision"

    def apply_glif_vision(
            self,
            glif_vision: GlifVision,
            image,
            model,
            weight,
            attn_mask=None,
            start_at=0.0,
            end_at=1.0,
            unfold_batch=False
    ):
        model = model.clone()
        # img is [B, H, W, C]
        # img is channels last, switch it to channels first (bs, c, w, h)
        image = image.permute(0, 3, 1, 2)
        # image = image.permute(0, 3, 2, 1)
        # run it through to get the kv list
        kv_list = glif_vision(image, attn_mask=attn_mask)

        # get the sigmas so we can shut off at the proper percent
        sigmas = calculate_sigmas(model.get_model_object("model_sampling"), 'simple', 1000)

        # determine the sigma at the percent decimal
        sigma_start = sigmas[math.floor(start_at * (len(sigmas) - 1))]
        sigma_end = sigmas[math.floor(end_at * (len(sigmas) - 1))]

        # hack the forward function to save the transformer options
        hack_forward_to_save_kwargs(model)

        patch_kwargs = {
            "number": 0,
            "weight": weight,
            "kv": None,
            "mask": attn_mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }

        block_idx = 0
        for id in range(19):  # 57 total blocks (19 mmdit, 38 single dit)
            patch_kwargs["kv"] = kv_list[block_idx]
            patch_kwargs["number"] = block_idx
            patch_kwargs = {**patch_kwargs}
            add_block_forward_patch(model, patch_kwargs, ("double_blocks", id), double_stream_forward)
            block_idx += 1

        for id in range(38):  # 57 total blocks (19 mmdit, 38 single dit)
            patch_kwargs["kv"] = kv_list[block_idx]
            patch_kwargs["number"] = block_idx
            patch_kwargs = {**patch_kwargs}
            add_block_forward_patch(model, patch_kwargs, ("single_blocks", id), single_stream_forward)
            block_idx += 1

        return (model,)


NODE_CLASS_MAPPINGS = {
    # Main Apply Nodes
    "GlifVisionApply": GlifVisionApply,
    "GlifVisionModelLoader": GlifVisionModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Main Apply Nodes
    "GlifVisionApply": "Glif Vision Apply",
    "GlifVisionModelLoader": "Glif Vision Model Loader",
}
