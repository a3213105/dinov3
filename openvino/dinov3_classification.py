#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 视你的层级调整
from ov_model_helper import DinoV3ClassificationWrapper, DinoV3LastHiddenStateWrapper
from ov_operator_async import DinoV3ClassificationModel
from utils import convert_and_load_ov_model, load_ov_base_model

@torch.inference_mode()
def convert_classification_pipeline(model, xml_path, pixel_values, pool: str, topk: int):
    DinoV3ClassificationWrapper(model, using_cls_token=True if pool == "cls" else False).convert_model(xml_path, pixel_values, topk)

@torch.inference_mode()
def convert_classification_base(model, xml_path, pixel_values):
    DinoV3LastHiddenStateWrapper(model).convert_model(xml_path, pixel_values)

@torch.inference_mode()
def classification(model, pixel_values, pool: str, topk: int = 3):
    # B, C, H, W = pixel_values.shape
    outputs = model(pixel_values=pixel_values)
    seq = outputs.last_hidden_state  # [B, N+1, C]

    if pool == "cls":
        feat = seq[:, 0, :]  # CLS
    else:
        feat = seq[:, 1:, :].mean(dim=1)  # mean over patches
    # L2-normalize for cosine similarity
    feat = torch.nn.functional.normalize(feat, p=2, dim=-1).squeeze(0)

    _, idxs = torch.topk(feat, k=topk, largest=True, sorted=True)
    return idxs.cpu().numpy()

@torch.inference_mode()
def classification_ov_pipe(model, pixel_values, pool, topk):
    idxs = model(pixel_values, topk)
    return idxs

def load_object_discovery_model(model_path, model_name):
    model = DinoV3ClassificationModel(model_path + "/" + model_name + ".xml")
    model.setup_model(stream_num = 1, bf16=True, f16=False)
    return model

@torch.inference_mode()
def do_classification(tp, model_path, model_name, model, pixel_values, pool: str, topk: int = 3, loop=1):
    if model is None:
        if tp == 'ov_all' :
            model = convert_and_load_ov_model(tp, model_path, model_name, "classification", convert_classification_pipeline, load_object_discovery_model, pixel_values, pool=pool, topk=topk)
        else :
            model = convert_and_load_ov_model(tp, model_path, model_name, "classification", convert_classification_base, load_ov_base_model, pixel_values, pool=pool, topk=topk)
    if tp == 'ov_all' :
        classification_fn = classification_ov_pipe
    else :
        classification_fn = classification
    idxs = classification_fn(model, pixel_values, pool, topk)
    # print(f"[INFO] Image Classification with TOP-{topk}: {idxs.tolist()}")
    if loop > 1:
        start = time.perf_counter()
        for _ in range(loop):
            classification_fn(model, pixel_values, pool, topk)
        duration = time.perf_counter() - start
        print(f"[INFO] Classification {tp} inference {model_name} loop={loop} times, average time: {duration/loop:.4f} seconds, {loop/duration:.3f} FPS")
    return idxs
