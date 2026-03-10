#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import torch
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 视你的层级调整
from ov_model_helper import DinoV3LastHiddenStateWrapper
from utils import colorize_mask, colorize_with_palette, normalize_to_uint8, convert_and_load_ov_model, load_ov_base_model

@torch.inference_mode()
def convert_segmentation_base(model, xml_path, pixel_values):
    DinoV3LastHiddenStateWrapper(model).convert_model(xml_path, pixel_values)

@torch.inference_mode()
def segmentation(model, pixel_values, kmeans, true_N):
    # Forward through DINOv3 backbone
    outputs = model(pixel_values=pixel_values)
    # DINOv3 HF returns last_hidden_state: [1, N+1, C]
    seq = outputs.last_hidden_state  # includes CLS at index 0
    # Remove CLS token
    seq = seq[:, 1:, :]  # [1, N, C]
    _, N_all, C = seq.shape
   
    if N_all == true_N + 1:
        seq = seq[:, 1:, :]
    elif N_all != true_N:
        if N_all > true_N:
            seq = seq[:, :true_N, :]
        else:
            last = seq[:, -1:, :]
            # repeat 到缺口长度: shape [1, true_N - N_all, C]
            pad = np.repeat(last, repeats=(true_N - N_all), axis=1)
            # 拼接: shape [1, true_N, C]
            seq = np.concatenate([seq, pad], axis=1)
    tokens = seq.reshape(true_N, C)    
    labels = kmeans.fit_predict(tokens)
    return labels

@torch.inference_mode()
def segmentation_postprocess(labels, H, W, gh, gw, outputname):
    seg_small = labels.reshape(gh, gw).astype(np.uint8)
    seg_full = Image.fromarray(seg_small, mode='L').resize((W, H), resample=Image.NEAREST)
    seg = np.array(seg_full)
    # Colorize & save
    color = colorize_mask(seg)
    out_img = Image.fromarray(color)
    out_img.save(outputname)
    # print(f"[INFO] Image Segmentation saved to {outputname}")

@torch.inference_mode()
def do_segmentation(tp, model_path, model_name, model, pixel_values, kmeans, k, outputname, loop=1):
    if model is None:
        model = convert_and_load_ov_model('ov', model_path, model_name, "segmentation", convert_segmentation_base, load_ov_base_model, pixel_values)

    patch_size = 32 if model.config.model_type == "dinov3_convnext" else model.config.patch_size
    _, _, H, W = pixel_values.shape
    gh = H // patch_size
    gw = W // patch_size
    true_N = gh * gw
    labels = segmentation(model, pixel_values, kmeans, true_N)
    segmentation_postprocess(labels, H, W, gh, gw, outputname)
    if loop > 1:
        start = time.perf_counter()
        if tp == 'ov_all' :
            for _ in range(loop):
                model(pixel_values)
        else :
            for _ in range(loop):
                segmentation(model, pixel_values, kmeans, true_N)
        duration = time.perf_counter() - start
        print(f"[INFO] Segmentation {tp} inference {model_name} loop={loop} times, average time: {duration/loop:.4f} seconds, {loop/duration:.3f} FPS")
