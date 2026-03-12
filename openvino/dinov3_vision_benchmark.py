#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 视你的层级调整
from utils import process_image, parser_args, load_hf_model, load_ov_base_model
from dinov3_segmentation import do_segmentation
from dinov3_classification import do_classification
from dinov3_embedding import do_embedding
from dinov3_object_discovery import do_object_discovery
from dinov3_depth import do_depth_pca

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parser_args()
    
    model_names = [ "dinov3-convnext-tiny-pretrain-lvd1689m",   #29M
                    "dinov3-convnext-small-pretrain-lvd1689m",  #50M
                    "dinov3-convnext-base-pretrain-lvd1689m",   #89M
                    "dinov3-convnext-large-pretrain-lvd1689m",  #198M
                    "dinov3-vits16-pretrain-lvd1689m",          #21M
                    "dinov3-vits16plus-pretrain-lvd1689m",      #29M
                    "dinov3-vitb16-pretrain-lvd1689m",          #86M
                    "dinov3-vitl16-pretrain-lvd1689m",          #300M
                    "dinov3-vith16plus-pretrain-lvd1689m",      #840M
                    "dinov3-vit7b16-pretrain-lvd1689m",         #6716M
                  ]

    # Load image
    image_pil = Image.open(args.image).convert("RGB")
    W, H = image_pil.size
    pixel_values = process_image(image_pil)
    for i, model_name in enumerate(model_names):
        # print(f"### processing model {i+1}/{len(model_names)}: {model_name}")
        segment_path = args.output + model_name + "_seg.png"
        mask_path= args.output + model_name + "_mask.png"
        overlay_path= args.output + model_name + "_overlay.png"
        depth_path = args.output + model_name + "_depth.png"
        color_path = args.output + model_name + "_depth_color.png"
        kmeans = KMeans(n_clusters=args.k, n_init='auto', random_state=0)
        kmeans_seg = KMeans(n_clusters=2, n_init=10, random_state=0)
        pca = PCA(n_components=1, random_state=0)
        if True :
            for tp in ['ov_all', 'ov', 'torch'] :
                try :
                    if tp == 'torch' :
                        model = load_hf_model(args.model_path, model_name)
                    elif tp == 'ov' :
                        model = load_ov_base_model(f'{args.model_path}_ov/{model_name}', model_name)
                    else :
                        model = None
                except Exception as e:
                    print(f"### Failed to load model {model_name} for {tp}: {e}")
                    if tp == 'torch' :
                        continue
                    else :
                        model = None
                        print(f"### Try to Convert model {model_name} to {tp} for benchmark...")
                    
                if args.tasks is None or "segmentation" in args.tasks:
                    do_segmentation(tp, args.model_path, model_name, model, pixel_values, kmeans_seg, args.k, segment_path, loop=args.loop)
                if args.tasks is None or "classification" in args.tasks:
                    do_classification(tp, args.model_path, model_name, model, pixel_values, pool="cls", topk=args.k, loop=args.loop)
                if args.tasks is None or "embedding" in args.tasks:
                    do_embedding(tp, args.model_path, model_name, model, pixel_values, loop=args.loop)
                if args.tasks is None or "object_discovery" in args.tasks:
                    do_object_discovery(tp, args.model_path, model_name, model, pixel_values, kmeans, image_pil, mask_path, overlay_path, loop=args.loop)
                if args.tasks is None or "depth" in args.tasks:
                    do_depth_pca(tp, args.model_path, model_name, model, pixel_values, pca, depth_path, color_path, loop=args.loop)

if __name__ == "__main__":
    main()

