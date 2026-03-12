#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils import colorize_with_palette, normalize_to_uint8
from PIL import Image, ImageFilter
import numpy as np
import torch
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 视你的层级调整
from ov_model_helper import DinoV3LastHiddenStateWrapper, DinoV3DepthWrapper
from utils import convert_and_load_ov_model, load_ov_pipe_model, load_ov_base_model

@torch.inference_mode()
def convert_depth_base(model, xml_path, pixel_values):
    DinoV3LastHiddenStateWrapper(model).convert_model(xml_path, pixel_values)
    
@torch.inference_mode()
def convert_depth_pipeline(model, xml_path, pixel_values):
    DinoV3DepthWrapper(model).convert_model(xml_path, pixel_values)

def depth_postprocess(depth_small: np.ndarray, H: int, W: int, depth_path: str, color_path: str):
    # 4) 上采样到输入尺寸 + 轻后处理（双边滤波）
    depth_F = Image.fromarray(depth_small.astype(np.float32), mode="F")
    depth_up_F = depth_F.resize((W, H), resample=Image.BICUBIC)
    # convert to 8-bit for smoothing/colorization
    depth_up = np.array(depth_up_F, dtype=np.float32)
    depth_u8 = normalize_to_uint8(depth_up)
    depth_img_L = Image.fromarray(depth_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=1.5))

    depth_img_L.save(depth_path)
    
    color_img = colorize_with_palette(np.array(depth_img_L))
    color_img.save(color_path)

    # print(f"[INFO] Image Depth saved to {depth_path} and {color_path}")

@torch.inference_mode()
def depth(model, pixel_values):
    _, _, H, W = pixel_values.shape
    outputs = model(pixel_values=pixel_values)
    last = outputs.last_hidden_state
    _, _, D = last.shape
    if model.config.model_type == "dinov3_convnext" :
        patch_size = 32  # for ConvNeXt
    else :
        patch_size = model.config.patch_size
    gh, gw = H // patch_size, W // patch_size
    patch_tokens = last[:, - (gh * gw):, :]          # 取最后 gh*gw 个 patch token
    pt = torch.nn.functional.normalize(patch_tokens.squeeze(0), dim=-1)  # (gh*gw, D)
    feats = pt.reshape(gh, gw, D).contiguous()
    return feats

@torch.inference_mode()
def depth_ov_pipe(model, pixel_values):
    feats = model(pixel_values=pixel_values)
    return feats

@torch.inference_mode()
def depth_pca_pipeline(model, pixel_values, pca, depth_fn):
    feats = depth_fn(model, pixel_values)
    Hf, Wf, D = feats.shape

    # —— 零样本相对深度（PCA）——
    flat = feats.reshape(-1, D)
    depth_small = pca.fit_transform(flat).reshape(Hf, Wf)
    # 方向性：用特征范数的全局相关性决定是否翻转（避免远近颠倒）
    norms = np.linalg.norm(flat, axis=-1).reshape(Hf, Wf)
    corr = np.corrcoef(depth_small.reshape(-1), norms.reshape(-1))[0, 1]
    if not np.isnan(corr) and corr < 0:
        depth_small = -depth_small
    return depth_small

@torch.inference_mode()
def depth_head_pipeline(model, pixel_values, head, depth_fn):
    feats = depth_fn(model, pixel_values)
    _, _, D = feats.shape
    fmap = feats.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,D,Hf,Wf)
    pred = head(fmap)                              # (1,1,Hf,Wf)
    depth_small = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
    return depth_small


@torch.inference_mode()
def do_depth_pca(tp, model_path, model_name, model, pixel_values, pca, depth_path: str, color_path: str, loop=1):
    if model is None:
        if tp == 'ov_all' :
            model = convert_and_load_ov_model(tp, model_path, model_name, "depth", convert_depth_pipeline, load_ov_pipe_model, pixel_values)
        else :
            model = convert_and_load_ov_model(tp, model_path, model_name, "depth", convert_depth_base, load_ov_base_model, pixel_values)
    if tp == 'ov_all' :
        depth_fn = depth_ov_pipe
    else :
        depth_fn = depth
    depth_small = depth_pca_pipeline(model, pixel_values, pca, depth_fn)
    _, _, H, W = pixel_values.shape
    depth_postprocess(depth_small, H, W, depth_path, color_path)
    if loop > 0:
        start = time.perf_counter()
        for _ in range(loop):
            depth_pca_pipeline(model, pixel_values, pca, depth_fn)
        duration = time.perf_counter() - start
        print(f"[INFO] DepthPCA {tp} inference {model_name} loop={loop} times, average time: {duration/loop:.4f} seconds, {loop/duration:.3f} FPS")

@torch.inference_mode()
def do_depth_head(tp, model_path, model_name, model, pixel_values, head, depth_path: str, color_path: str, loop=1):
    if model is None:
        if tp == 'ov_all' :
            model = convert_and_load_ov_model(tp, model_path, model_name, "depth", convert_depth_pipeline, load_ov_pipe_model, pixel_values)
        else :
            model = convert_and_load_ov_model(tp, model_path, model_name, "depth", convert_depth_base, load_ov_base_model, pixel_values)
    if tp == 'ov_all' :
        depth_fn = depth_ov_pipe
    else :
        depth_fn = depth
    depth_small = depth_head_pipeline(model, pixel_values, head, depth_fn)
    _, _, H, W = pixel_values.shape
    depth_postprocess(depth_small, H, W, depth_path, color_path)
    if loop > 0:
        start = time.perf_counter()
        for _ in range(loop):
           depth_head_pipeline(model, pixel_values, head, depth_fn)
        duration = time.perf_counter() - start
        print(f"[INFO] DepthHead {tp} inference {model_name} loop={loop} times, average time: {duration/loop:.4f} seconds, {loop/duration:.3f} FPS")