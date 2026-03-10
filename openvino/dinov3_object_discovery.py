#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
from PIL import Image
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 视你的层级调整
from ov_model_helper import DinoV3ObjectDiscoveryWrapper, DinoV3LastHiddenStateWrapper
from ov_operator_async import DinoV3ObjectDiscoveryModel
from utils import convert_and_load_ov_model, load_ov_base_model

@torch.inference_mode()
def convert_object_discovery_pipeline(model, xml_path, pixel_values):
    DinoV3ObjectDiscoveryWrapper(model).convert_model(xml_path, pixel_values)

@torch.inference_mode()
def convert_object_discovery_base(model, xml_path, pixel_values):
    DinoV3LastHiddenStateWrapper(model).convert_model(xml_path, pixel_values)

@torch.inference_mode()
def object_discovery(model, pixel_values):
    outputs = model(pixel_values)

    seq = outputs.last_hidden_state  # (1, N, D)

    # separate tokens
    cls_token = seq[:, 0:1, :]              # (1,1,D)

    if model.config.model_type == "dinov3_convnext" :
        patch_tokens = seq[:, 1:, :]            # (1, gh*gw, D)
    else :
        patch_tokens = seq[:, 5:, :]            # (1, gh*gw, D)

    # normalize embeddings
    pt = torch.nn.functional.normalize(patch_tokens.squeeze(0), dim=-1)  # (gh*gw, D)
    cls = torch.nn.functional.normalize(cls_token.squeeze(0), dim=-1)    # (1, D)
    # Heuristic: compute avg cosine sim to CLS per cluster; choose higher as foreground
    sim = (pt @ cls.t()).squeeze(-1)                   # (gh*gw,)
    pt = pt.cpu().numpy()
    sim = sim.cpu().numpy()
    return pt, sim

@torch.inference_mode()
def object_discovery_ov(model, pixel_values):
    pt, sim = model(pixel_values=pixel_values)
    return pt, sim

@torch.inference_mode()
def object_discovery_pipeline(model, kmeans, pixel_values, patch_size, object_discovery_fn):
    pt, sim = object_discovery_fn(model, pixel_values)

    labels = kmeans.fit_predict(pt)      # (gh*gw,)

    # labels: (gh*gw,)
    labels_np = labels.astype(np.int64)

    # 计算 fg0 / fg1
    if np.any(labels_np == 0):
        fg0 = sim[labels_np == 0].mean()
    else:
        fg0 = -1e9

    if np.any(labels_np == 1):
        fg1 = sim[labels_np == 1].mean()
    else:
        fg1 = -1e9

    fg_label = int(fg1 > fg0)

    # 尺寸
    H, W = pixel_values.shape[-2:]
    gh, gw = H // patch_size, W // patch_size

    # (1,1,gh,gw)
    mask_small = (labels_np == fg_label).reshape(1, 1, gh, gw)
    mask_small = torch.from_numpy(mask_small).float()

    mask_up = torch.nn.functional.interpolate(mask_small, size=(H, W), mode="bilinear", align_corners=False)  # (1,1,H,W)
    mask_smooth = torch.nn.functional.avg_pool2d(mask_up, kernel_size=3, stride=1, padding=1)
    mask_bin = (mask_smooth > 0.5).float()  # binary
    return mask_bin

@torch.inference_mode()
def object_discovery_postprocess(mask_bin, img, H, W, mask_path, overlay_path):
    # 5) Save outputs
    mask_np = (mask_bin.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(mask_path)

    # overlay
    img_rs = img.resize((W, H))
    img_np = np.array(img_rs).astype(np.float32)
    overlay = img_np.copy()
    # red overlay on fg
    overlay[mask_np > 0] = 0.6 * overlay[mask_np > 0] + 0.4 * np.array([255, 0, 0], dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_path)
    # print(f"[INFO] Image Object Discovery mask saved to {mask_path} and overlay saved to {overlay_path}")

def load_object_discovery_model(model_path, model_name):
    model = DinoV3ObjectDiscoveryModel(model_path + "/" + model_name + ".xml")
    model.setup_model(stream_num = 1, bf16=True, f16=False)
    return model

@torch.inference_mode()
def do_object_discovery(tp, model_path, model_name, model, pixel_values, kmeans, img, mask_path, overlay_path, loop=1):
    if model is None:
        if tp == 'ov_all' :
            model = convert_and_load_ov_model(tp, model_path, model_name, "object_discovery", convert_object_discovery_pipeline, load_object_discovery_model, pixel_values)
        else :
            model = convert_and_load_ov_model(tp, model_path, model_name, "object_discovery", convert_object_discovery_base, load_ov_base_model, pixel_values)
    if tp == 'ov_all' :
        object_discovery_fn = object_discovery_ov
    else :
        object_discovery_fn = object_discovery

    _, _, H, W = pixel_values.shape
    if model.config.model_type == "dinov3_convnext" :
        patch_size = 32  # for ConvNeXt
    else :
        patch_size = model.config.patch_size

    mask_bin = object_discovery_pipeline(model, kmeans, pixel_values, patch_size, object_discovery_fn)
    object_discovery_postprocess(mask_bin, img, H, W, mask_path, overlay_path)
    if loop > 1:
        start = time.perf_counter()
        for _ in range(loop):
            object_discovery_pipeline(model, kmeans, pixel_values, patch_size, object_discovery_fn)
        duration = time.perf_counter() - start
        print(f"[INFO] Object_Discovery {tp} inference {model_name} loop={loop} times, average time: {duration/loop:.4f} seconds, {loop/duration:.3f} FPS")
