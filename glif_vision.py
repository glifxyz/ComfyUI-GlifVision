import os

import torch
import torch.nn as nn
import comfy.model_management
from transformers import SiglipImageProcessor, SiglipVisionModel, SiglipVisionConfig


class KVPreprocessorModule(nn.Module):
    def __init__(self, input_size, output_size, has_bias=True):
        super(KVPreprocessorModule, self).__init__()
        self.to_k_adapter = nn.Linear(input_size, output_size, bias=has_bias)
        self.to_v_adapter = nn.Linear(input_size, output_size, bias=has_bias)

    def forward(self, x):
        k = self.to_k_adapter(x)
        v = self.to_v_adapter(x)
        return k, v


class GlifVisionKVPreprocessor(nn.Module):
    def __init__(self, state_dict=None):
        super(GlifVisionKVPreprocessor, self).__init__()
        adapter_modules = []
        i = 0
        while True:
            if f"adapter_modules.{i}.to_k_adapter.weight" in state_dict:
                adapter_modules.append(KVPreprocessorModule(
                    state_dict[f"adapter_modules.{i}.to_k_adapter.weight"].shape[1],
                    state_dict[f"adapter_modules.{i}.to_k_adapter.weight"].shape[0],
                    has_bias="adapter_modules.{i}.to_k_adapter.bias" in state_dict
                ))
                i += 1
            else:
                break
        print(f"Found {len(adapter_modules)} glif vision adapter modules")

        self.adapter_modules = nn.ModuleList(adapter_modules)

        # load state_dict
        self.load_state_dict(state_dict)
        pass

    def forward(self, x):
        kv_list = []
        for adapter in self.adapter_modules:
            k, v = adapter(x)
            kv_list.append((k, v))
        return kv_list


class GlifVision(nn.Module):
    def __init__(self, state_dict=None):
        super(GlifVision, self).__init__()
        self.load_device = comfy.model_management.text_encoder_device()
        self.offload_device = comfy.model_management.text_encoder_offload_device()
        self.clip_layer = 'last_hidden_state'
        if state_dict is None:
            raise ValueError("state_dict is required for GlifVision")

        # seperate out state_dict into different parts
        dv_state_dict = {}
        vision_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("dvadapter."):
                dv_state_dict[k[len("dvadapter."):]] = v
            elif k.startswith("vision_encoder."):
                vision_state_dict[k[len("vision_encoder."):]] = v
            else:
                print(f"Unknown key: {k}")

        self.dvadapter: GlifVisionKVPreprocessor = GlifVisionKVPreprocessor(dv_state_dict)

        siglip_config_path = os.path.join(os.path.dirname(__file__), "configs", "siglip-base-patch16-512.json")
        siglip_processor_config_path = os.path.join(
            os.path.dirname(__file__), "configs",
            "siglip-base-patch16-512_preprocessor.json"
        )
        siglip_config = SiglipVisionConfig.from_pretrained(siglip_config_path)

        self.image_processor = SiglipImageProcessor.from_pretrained(siglip_processor_config_path)
        # self.vision_encoder: SiglipVisionModel = SiglipVisionModel(siglip_config)
        self.vision_encoder: SiglipVisionModel = SiglipVisionModel.from_pretrained(
            None,
            config=siglip_config,
            state_dict=vision_state_dict
        )
        self.vision_encoder.eval()
        self.load_dtype = torch.float32
        if comfy.model_management.should_use_fp16(self.load_device):
            self.load_dtype = torch.float16
        if comfy.model_management.should_use_bf16(self.load_device):
            self.load_dtype = torch.bfloat16
        self.to(self.offload_device, dtype=self.load_dtype)

    def forward(self, tensors_0_1, attn_mask=None):
        self.to(self.load_device)
        if tensors_0_1.min() < -0.3 or tensors_0_1.max() > 1.3:
            raise ValueError("image tensor values must be between 0 and 1. Got min: {}, max: {}".format(
                tensors_0_1.min(), tensors_0_1.max()
            ))
        clip_image = self.image_processor(
            images=tensors_0_1,
            return_tensors="pt",
            do_resize=True,
            do_rescale=False,
        ).pixel_values
        clip_image = clip_image.to(self.load_device, dtype=self.load_dtype)
        embeds = self.vision_encoder(
            clip_image,
            output_hidden_states=True
        )
        if self.clip_layer == 'penultimate_hidden_states':
            embeds = embeds.hidden_states[-2]
        elif self.clip_layer == 'last_hidden_state':
            embeds = embeds.hidden_states[-1]
        elif self.clip_layer == 'image_embeds':
            embeds = embeds.image_embeds
        else:
            raise ValueError(f"unknown clip layer: {self.clip_layer}")

        # todo fix attn masking, this doesnt work well
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            # repeat ch to 3
            if attn_mask.shape[1] == 1:
                attn_mask = attn_mask.repeat(1, 3, 1, 1)
            attn_mask = self.image_processor(
                images=attn_mask.float(),
                return_tensors="pt",
                do_resize=True,
                do_rescale=False,
                do_normalize=False,
            ).pixel_values
            attn_mask = attn_mask.to(self.load_device, dtype=self.load_dtype)
            # Average across the color channels
            mask = attn_mask.mean(dim=1, keepdim=True)  # (bs, 1, 512, 512)

            # Resize to match the number of patches
            mask = torch.nn.functional.interpolate(
                mask, size=(32, 32), mode='bicubic', align_corners=False
            )  # (bs, 1, 32, 32)

            # Flatten the spatial dimensions
            mask = mask.flatten(2)  # (bs, 1, 1024)

            # Transpose to get the right shape
            mask = mask.transpose(1, 2)  # (bs, 1024, 1)
            # determine scale to make a mean of 1
            embeds = embeds * mask
            # print("Attention mask is not implemented yet")

        # preprocess key and value
        kv_list = self.dvadapter(embeds)

        # todo handle this for unfolding
        if embeds.shape[0] > 1:
            kv_list = [(k.mean(dim=0, keepdim=True), v.mean(dim=0, keepdim=True)) for k, v in kv_list]
        self.to(self.offload_device)
        return kv_list
