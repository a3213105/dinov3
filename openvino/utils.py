#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import argparse # type: ignore
import numpy as np
import os
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoModel
import torch
from torchvision import transforms
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 视你的层级调整
from ov_operator_async import DinoV3BaseModel, DinoV3PipeModel
from ov_model_helper import DinoV3BaseWrapper

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

# ---------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------
def colorize_mask(mask):
    np.random.seed(0)
    K = mask.max() + 1
    palette = np.random.randint(0, 255, (K, 3), dtype=np.uint8)
    return palette[mask]

def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Robust linear stretch to [0,255] using 2%~98% percentiles."""
    x = x.astype(np.float32)
    mn, mx = np.percentile(x, 2.0), np.percentile(x, 98.0)
    if mx <= mn:
        mn, mx = x.min(), x.max()
    y = np.clip((x - mn) / (mx - mn + 1e-8), 0, 1)
    return (y * 255.0 + 0.5).astype(np.uint8)

def make_jet_palette() -> list:
    """
    Create a simple 'jet-like' palette (256 * 3 list) for PIL 'P' images.
    Blue -> Cyan -> Yellow -> Red gradient, approximated in pure Python.
    """
    pal = []
    for i in range(256):
        x = i / 255.0
        # piecewise gradient (approx jet)
        if x < 0.25:  # Blue -> Cyan
            r = 0
            g = int(4 * x * 255)
            b = 255
        elif x < 0.5:  # Cyan -> Green
            r = 0
            g = 255
            b = int(255 - 4 * (x - 0.25) * 255)
        elif x < 0.75:  # Green -> Yellow
            r = int(4 * (x - 0.5) * 255)
            g = 255
            b = 0
        else:  # Yellow -> Red
            r = 255
            g = int(255 - 4 * (x - 0.75) * 255)
            b = 0
        pal.extend([r, g, b])
    return pal

def colorize_with_palette(gray_u8: np.ndarray) -> Image.Image:
    """Apply our jet-like palette via PIL palette mode (no OpenCV)."""
    img_L = Image.fromarray(gray_u8, mode="L")
    img_P = img_L.convert("P")
    img_P.putpalette(make_jet_palette())
    return img_P.convert("RGB")

def process_image(image_pil):
    # W, H = image_pil.size
    tfm = build_transform()
    # Convert to tensor
    x = tfm(image_pil).unsqueeze(0)  # [1,3,H,W]
    return x

def load_hf_model(model_path, model_name):
    model = AutoModel.from_pretrained(model_path + "/" + model_name,
                                          dtype=torch.float, device_map="cpu",
                                        #   attn_implementation="sdpa",
                                         )
    # model = model.bfloat16().eval()
    model = model.eval()
    return model

@torch.inference_mode()
def convert_ov_model(model, xml_path, pixel_values):
    dino_wrapper = DinoV3BaseWrapper(model)
    dino_wrapper.convert_model(xml_path, pixel_values)

def load_ov_base_model(model_path, model_name):
    model = DinoV3BaseModel(model_path + "/" + model_name + ".xml")
    model.setup_model(stream_num = 1, bf16=True, f16=False)
    return model

def load_ov_pipe_model(model_path, model_name):
    model = DinoV3PipeModel(model_path + "/" + model_name + ".xml")
    model.setup_model(stream_num = 1, bf16=True, f16=False)
    return model

def convert_and_load_ov_model(tp, model_path, model_name, workload, convert_ov_model, load_ov_model, pixel_values, **xargs):
    output_path = f'{model_path}_{tp}/{model_name}'
    if tp == "ov_all" and workload is not None :
        output_name = f'{model_name}_{workload}'
    else :
        output_name: str = f'{model_name}'
    xml_file = f'{output_path}/{output_name}.xml'
    # print(f"### converting {tp} model {model_name} from {model_path} to {xml_file} for {workload} workload...")
    if not os.path.exists(xml_file):
        model_pt = load_hf_model(model_path, model_name)
        convert_ov_model(model_pt, xml_file, pixel_values, **xargs)
    if load_ov_model is None :
        return load_ov_base_model(output_path, output_name)
    return load_ov_model(output_path, output_name)

def parser_args() :
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="../../000000039769.jpg", help="Input image path")
    ap.add_argument("--k", type=int, default=5, help="Number of clusters")
    ap.add_argument("--show", action="store_true", help="Show visualization")
    ap.add_argument("--output", type=str, default="../../outputs/", help="Output directory")
    ap.add_argument("--model_path", type=str, default="../../models")
    ap.add_argument("--loop", type=int, default=1, help="benchmark loop times")
    ap.add_argument("--tasks", type=str, nargs="+", default=None, help="Tasks to run: segmentation, classification, embedding, object_discovery, depth")
    return ap.parse_args()
