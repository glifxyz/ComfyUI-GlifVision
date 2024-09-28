import torch
from torch import Tensor

from comfy.ldm.flux.math import attention, apply_rope
from comfy.ldm.modules.attention import optimized_attention


def double_stream_forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, extra_options=None):
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    # run actual attention
    # attn = attention(torch.cat((txt_q, img_q), dim=2),
    #                  torch.cat((txt_k, img_k), dim=2),
    #                  torch.cat((txt_v, img_v), dim=2), pe=pe)
    q = torch.cat([txt_q, img_q], dim=2)
    k = torch.cat([txt_k, img_k], dim=2)
    v = torch.cat([txt_v, img_v], dim=2)
    q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    attn = optimized_attention(q, k, v, heads, skip_reshape=True)

    # glif vision patch
    if hasattr(self, "glif_vision_patch_kwargs") and self.glif_vision_patch_kwargs is not None:
        sigma = 999999999.9
        if hasattr(self, 'transformer_options') and 'sigmas' in self.transformer_options:
            sigma = self.transformer_options['sigmas'].detach().cpu()[0].item()
        patch_kwargs = self.glif_vision_patch_kwargs
        sigma_start = patch_kwargs['sigma_start']
        sigma_end = patch_kwargs['sigma_end']
        if sigma <= sigma_start and sigma >= sigma_end:
            if 'kv' in patch_kwargs and patch_kwargs['kv'] is not None:
                kv = patch_kwargs['kv']
                adapter_k, adapter_v = kv
                batch_size = q.shape[0]
                inner_dim = adapter_k.shape[-1]
                head_dim = inner_dim // heads  # 128
                adapter_k = adapter_k.view(batch_size, -1, heads, head_dim).transpose(1, 2)
                adapter_v = adapter_v.view(batch_size, -1, heads, head_dim).transpose(1, 2)
                adapter_attn = optimized_attention(q, adapter_k, adapter_v, heads, skip_reshape=True)
                attn = attn + (adapter_attn * self.glif_vision_patch_kwargs['weight'])

    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

    # calculate the img bloks
    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

    # calculate the txt bloks
    txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt

def single_stream_forward(self, x: Tensor, vec: Tensor, pe: Tensor, extra_options=None) -> Tensor:
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = self.norm(q, k, v)

    # compute attention
    q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    attn = optimized_attention(q, k, v, heads, skip_reshape=True)

    # glif vision patch
    if hasattr(self, "glif_vision_patch_kwargs") and self.glif_vision_patch_kwargs is not None:
        sigma = 999999999.9
        if hasattr(self, 'transformer_options') and 'sigmas' in self.transformer_options:
            sigma = self.transformer_options['sigmas'].detach().cpu()[0].item()
        patch_kwargs = self.glif_vision_patch_kwargs
        sigma_start = patch_kwargs['sigma_start']
        sigma_end = patch_kwargs['sigma_end']

        if sigma <= sigma_start and sigma >= sigma_end:
            if 'kv' in patch_kwargs and patch_kwargs['kv'] is not None:
                kv = patch_kwargs['kv']
                adapter_k, adapter_v = kv
                batch_size = q.shape[0]
                inner_dim = adapter_k.shape[-1]
                head_dim = inner_dim // heads  # 128
                adapter_k = adapter_k.view(batch_size, -1, heads, head_dim).transpose(1, 2)
                adapter_v = adapter_v.view(batch_size, -1, heads, head_dim).transpose(1, 2)
                adapter_attn = optimized_attention(q, adapter_k, adapter_v, heads, skip_reshape=True)
                attn = attn + (adapter_attn * self.glif_vision_patch_kwargs['weight'])

    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += mod.gate * output

    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x
