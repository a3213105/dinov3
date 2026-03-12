#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 视你的层级调整
from ov_model_helper import DinoV3EmbeddingWrapper, DinoV3LastHiddenStateWrapper
from utils import convert_and_load_ov_model, load_ov_pipe_model, load_ov_base_model

@torch.inference_mode()
def convert_embedding_pipeline(model, xml_path, pixel_values):
    DinoV3EmbeddingWrapper(model).convert_model(xml_path, pixel_values)

@torch.inference_mode()
def convert_embedding_base(model, xml_path, pixel_values):
    DinoV3LastHiddenStateWrapper(model).convert_model(xml_path, pixel_values)

@torch.inference_mode()
def embedding(model, pixel_values) -> np.ndarray:
    outputs = model(pixel_values=pixel_values)
    last = outputs.last_hidden_state  # (B, N, D)
    feats = last.mean(dim=1)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats

@torch.inference_mode()
def embedding_ov_pipe(model, pixel_values) -> np.ndarray:
    feats = model(pixel_values=pixel_values)
    return feats
 
@torch.inference_mode()
def do_embedding(tp, model_path, model_name, model, pixel_values, loop) -> np.ndarray:
    if model is None:
        if tp == 'ov_all' :
            model = convert_and_load_ov_model(tp, model_path, model_name, "embedding", convert_embedding_pipeline, load_ov_pipe_model, pixel_values)
        else :
            model = convert_and_load_ov_model(tp, model_path, model_name, "embedding", convert_embedding_base, load_ov_base_model, pixel_values)
    if tp == 'ov_all' :
        embedding_fn = embedding_ov_pipe
    else :
        embedding_fn = embedding

    feats = embedding_fn(model, pixel_values)
    # print(f"[INFO] Image embedding vector size: {feats.shape[1]}")
    if loop > 0:
        start = time.perf_counter()
        for _ in range(loop):
            embedding_fn(model, pixel_values)
        duration = time.perf_counter() - start
        print(f"[INFO] Embedding {tp} inference {model_name} loop={loop} times, average time: {duration/loop:.4f} seconds, {loop/duration:.3f} FPS")
    return feats
