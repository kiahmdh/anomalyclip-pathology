"""
plip.py — load the PLIP pathology VLM (vinid/plip) into the *same* AnomalyCLIP
pipeline used for CLIP.

PLIP ships in HuggingFace transformers format (a CLIPModel, ViT-B/32). This module
converts its weights into the OpenAI-CLIP state-dict layout that AnomalyCLIP_lib's
build_model expects, caches it as a .pt, and then hands it to the SAME
AnomalyCLIP_lib.load(...). The returned model therefore has the identical API as the
CLIP backbone (encode_image with features_list+DPAM_layer, encode_text_learn,
visual.DAPM_replace, ...), so nothing else in train.py / eval has to change except
choosing this loader.

Usage in code:
    import plip
    model, _ = plip.load(device=device, design_details=AnomalyCLIP_parameters)

One-off build + sanity check:
    python plip.py            # converts, saves, and prints HF-vs-converted cosine
"""
import os
import re
import torch

import AnomalyCLIP_lib

HF_NAME = "vinid/plip"


def _default_ckpt():
    return os.path.expanduser("~/.cache/clip/plip_openai.pt")


def _n_layers(hf_sd, stem):
    idxs = set()
    for k in hf_sd:
        m = re.match(rf"{re.escape(stem)}\.encoder\.layers\.(\d+)\.", k)
        if m:
            idxs.add(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 0


def _convert_block(hf_sd, hf_prefix, oai_prefix, sd):
    """One transformer layer: HF (separate q/k/v) -> OpenAI (fused in_proj)."""
    qw = hf_sd[f"{hf_prefix}.self_attn.q_proj.weight"]
    kw = hf_sd[f"{hf_prefix}.self_attn.k_proj.weight"]
    vw = hf_sd[f"{hf_prefix}.self_attn.v_proj.weight"]
    qb = hf_sd[f"{hf_prefix}.self_attn.q_proj.bias"]
    kb = hf_sd[f"{hf_prefix}.self_attn.k_proj.bias"]
    vb = hf_sd[f"{hf_prefix}.self_attn.v_proj.bias"]
    sd[f"{oai_prefix}.attn.in_proj_weight"] = torch.cat([qw, kw, vw], dim=0)
    sd[f"{oai_prefix}.attn.in_proj_bias"] = torch.cat([qb, kb, vb], dim=0)
    sd[f"{oai_prefix}.attn.out_proj.weight"] = hf_sd[f"{hf_prefix}.self_attn.out_proj.weight"]
    sd[f"{oai_prefix}.attn.out_proj.bias"] = hf_sd[f"{hf_prefix}.self_attn.out_proj.bias"]
    sd[f"{oai_prefix}.ln_1.weight"] = hf_sd[f"{hf_prefix}.layer_norm1.weight"]
    sd[f"{oai_prefix}.ln_1.bias"] = hf_sd[f"{hf_prefix}.layer_norm1.bias"]
    sd[f"{oai_prefix}.ln_2.weight"] = hf_sd[f"{hf_prefix}.layer_norm2.weight"]
    sd[f"{oai_prefix}.ln_2.bias"] = hf_sd[f"{hf_prefix}.layer_norm2.bias"]
    sd[f"{oai_prefix}.mlp.c_fc.weight"] = hf_sd[f"{hf_prefix}.mlp.fc1.weight"]
    sd[f"{oai_prefix}.mlp.c_fc.bias"] = hf_sd[f"{hf_prefix}.mlp.fc1.bias"]
    sd[f"{oai_prefix}.mlp.c_proj.weight"] = hf_sd[f"{hf_prefix}.mlp.fc2.weight"]
    sd[f"{oai_prefix}.mlp.c_proj.bias"] = hf_sd[f"{hf_prefix}.mlp.fc2.bias"]


def convert_hf_to_openai(hf_model):
    """transformers CLIPModel state_dict -> OpenAI-CLIP state_dict (keys build_model expects)."""
    hf = hf_model.state_dict()
    sd = {}

    # ---------- text tower ----------
    sd["token_embedding.weight"] = hf["text_model.embeddings.token_embedding.weight"]
    sd["positional_embedding"] = hf["text_model.embeddings.position_embedding.weight"]
    for i in range(_n_layers(hf, "text_model")):
        _convert_block(hf, f"text_model.encoder.layers.{i}", f"transformer.resblocks.{i}", sd)
    sd["ln_final.weight"] = hf["text_model.final_layer_norm.weight"]
    sd["ln_final.bias"] = hf["text_model.final_layer_norm.bias"]
    sd["text_projection"] = hf["text_projection.weight"].t().contiguous()

    # ---------- vision tower ----------
    sd["visual.class_embedding"] = hf["vision_model.embeddings.class_embedding"]
    sd["visual.positional_embedding"] = hf["vision_model.embeddings.position_embedding.weight"]
    sd["visual.conv1.weight"] = hf["vision_model.embeddings.patch_embedding.weight"]
    pre = "vision_model.pre_layrnorm" if "vision_model.pre_layrnorm.weight" in hf else "vision_model.pre_layernorm"
    sd["visual.ln_pre.weight"] = hf[f"{pre}.weight"]
    sd["visual.ln_pre.bias"] = hf[f"{pre}.bias"]
    for i in range(_n_layers(hf, "vision_model")):
        _convert_block(hf, f"vision_model.encoder.layers.{i}", f"visual.transformer.resblocks.{i}", sd)
    sd["visual.ln_post.weight"] = hf["vision_model.post_layernorm.weight"]
    sd["visual.ln_post.bias"] = hf["vision_model.post_layernorm.bias"]
    sd["visual.proj"] = hf["visual_projection.weight"].t().contiguous()

    # ---------- logit scale ----------
    sd["logit_scale"] = hf["logit_scale"]
    return sd


def export_openai_checkpoint(out_path=None, hf_name=HF_NAME, force=False):
    """Download PLIP from HF, convert, and cache as a .pt. Returns the path."""
    out_path = out_path or _default_ckpt()
    if os.path.isfile(out_path) and not force:
        return out_path
    from transformers import CLIPModel
    hf_model = CLIPModel.from_pretrained(hf_name)
    sd = convert_hf_to_openai(hf_model)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(sd, out_path)
    print(f"[plip] wrote OpenAI-format checkpoint -> {out_path}  ({len(sd)} tensors)")
    return out_path


def load(device="cuda", design_details=None, hf_name=HF_NAME, ckpt_path=None):
    """Drop-in replacement for AnomalyCLIP_lib.load('ViT-L/14@336px', ...), but PLIP."""
    ckpt_path = export_openai_checkpoint(ckpt_path, hf_name)
    return AnomalyCLIP_lib.load(ckpt_path, device=device, design_details=design_details)


def _sanity_check():
    """Compare HF PLIP vs the converted model on one image+text. Image cosine must be ~1.0."""
    import numpy as np
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = export_openai_checkpoint()

    hf = CLIPModel.from_pretrained(HF_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(HF_NAME)
    model, _ = AnomalyCLIP_lib.load(path, device=device, design_details=None)
    model = model.float().eval()

    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    inp = proc(text=["an H&E image of lymph node tissue"], images=img,
        return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)

    def cos(a, b):
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        return float((a * b).sum(-1).mean())

    with torch.no_grad():
        hf_img = hf.get_image_features(pixel_values=inp["pixel_values"])
        oai_img = model.encode_image(inp["pixel_values"])
        if oai_img.dim() == 3:
            oai_img = oai_img[:, 0, :]   # this fork returns all tokens; token 0 is the global (CLS) embedding
        hf_txt = hf.get_text_features(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"))
        oai_txt = model.encode_text(inp["input_ids"])

    print(f"[plip] image-embedding cosine (HF vs converted): {cos(hf_img, oai_img):.4f}  (want ~1.0)")
    print(f"[plip] text-embedding  cosine (HF vs converted): {cos(hf_txt, oai_txt):.4f}  (want ~1.0)")
    print("[plip] if image cosine is ~1.0 the vision conversion is correct.")


if __name__ == "__main__":
    _sanity_check()

