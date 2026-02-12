#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUES200 OverLoCK-T LRFR prediction script.

Usage:
    python predict_sues200.py \
        --model_path ./checkpoints/.../best_score.pth \
        --dataset SUES200-S2D \
        --enable_geometric_rerank \
        --geometric_rerank_top_k 5

    python predict_sues200.py \
        --model_path ./checkpoints/.../best_score.pth \
        --dataset SUES200-D2S \
        --enable_geometric_rerank \
        --geometric_rerank_top_k 5
"""

import os
import gc
import random
import argparse
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.model_overlock import GeoModelOverLoCK
from overlock_t_config import get_config
from utils.trainer import evaluate, compute_mAP
from datasets.sues200 import SUES200DatasetEval, get_transforms


@dataclass
class OverLoCKConfiguration:
    """OverLoCK-T SUES200 prediction config."""
    # Model
    backbone: str = "overlock_t"
    attention: str = "GSRA"
    aggregation: str = "SALAD"

    num_channels: int = 640
    img_size: int = 512
    num_clusters: int = 128
    cluster_dim: int = 64

    fusion_mode: str = "main_only_no_compress"
    pretrained_path: str = None

    seed: int = 1
    verbose: bool = True

    # Eval
    batch_size_eval: int = 128
    eval_gallery_n: int = -1
    normalize_features: bool = True

    # Dataset
    dataset: str = "SUES200-D2S"
    data_folder: str = "/media/hk/soft/zx/GRSL-paper/sues200"
    drone_height: int = 150
    train_split: float = 0.7
    split_seed: int = 42

    # System
    num_workers: int = 0 if os.name == "nt" else 14
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="SUES200 OverLoCK-T LRFR prediction")

    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/.../best_score.pth",
        help="Path to trained checkpoint",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="SUES200-D2S",
        choices=["SUES200-D2S", "SUES200-S2D"],
        help="Dataset direction",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/media/hk/soft/zx/GRSL-paper/sues200",
        help="SUES200 dataset root folder",
    )
    parser.add_argument(
        "--drone_height",
        type=int,
        default=150,
        choices=[150, 200, 250, 300],
        help="Drone height folder",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.7,
        help="Train split ratio used in training",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed used for train/test split",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="overlock_t",
        choices=["overlock_t", "overlock_t_24x24"],
        help="OverLoCK backbone version",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="main_only_no_compress",
        choices=["main_only_no_compress", "no_compress"],
        help="OverLoCK fusion mode",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="GSRA",
        choices=["GSRA", "GSRA_overlock", "none"],
        help="Attention module type",
    )
    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="SALAD",
        choices=["SALAD", "SALAD_enhanced"],
        help="Aggregation module type",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Input image size",
    )
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--eval_gallery_n",
        type=int,
        default=-1,
        help="Gallery size for evaluation (-1 for all)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override dataloader workers",
    )

    # Geometric re-ranking
    parser.add_argument(
        "--enable_geometric_rerank",
        action="store_true",
        help="Enable geometric re-ranking",
    )
    parser.add_argument(
        "--geometric_rerank_top_k",
        type=int,
        default=10,
        help="Top-K candidates for geometric re-ranking",
    )

    # Visualization
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Save retrieval visualization results",
    )
    parser.add_argument(
        "--vis_output_dir",
        type=str,
        default="./visualization",
        help="Visualization output folder",
    )
    parser.add_argument(
        "--vis_top_k",
        type=int,
        default=3,
        help="Top-K for visualization",
    )

    return parser.parse_args()


def setup_overlock_config(config: OverLoCKConfiguration) -> OverLoCKConfiguration:
    """Setup OverLoCK-T specific config and validate."""
    print("=== OverLoCK-T config setup ===")

    if config.backbone == "overlock_t_24x24":
        if config.fusion_mode == "main_only_no_compress":
            overlock_config = get_config("overlock_main_only_no_compress")
        elif config.fusion_mode == "no_compress":
            overlock_config = get_config("overlock_24x24")
        else:
            raise ValueError(f"Unsupported fusion_mode: {config.fusion_mode}")
        print("  Using 24x24 backbone")
    elif config.fusion_mode == "main_only_no_compress":
        overlock_config = get_config("overlock_main_only_no_compress")
    elif config.fusion_mode == "no_compress":
        overlock_config = get_config("overlock_no_compress")
    else:
        raise ValueError(f"Unsupported fusion_mode: {config.fusion_mode}")

    user_backbone = config.backbone
    config.pretrained_path = overlock_config.get("pretrained_path")
    config.backbone = user_backbone

    if config.backbone == "overlock_t_24x24":
        if config.fusion_mode == "main_only_no_compress":
            config.num_channels = 640
        elif config.fusion_mode == "no_compress":
            config.num_channels = 1152
        else:
            raise ValueError(f"Unsupported fusion_mode: {config.fusion_mode}")
    elif config.backbone == "overlock_t":
        if config.fusion_mode == "main_only_no_compress":
            config.num_channels = 640
        elif config.fusion_mode == "no_compress":
            config.num_channels = 1152
        else:
            raise ValueError(f"Unsupported fusion_mode: {config.fusion_mode}")
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")

    print(f"  Backbone: {config.backbone}")
    print(f"  Fusion mode: {config.fusion_mode}")
    print(f"  Output dim: {config.num_channels}")
    print(f"  Input size: {config.img_size}x{config.img_size}")
    print(f"  Pretrained path: {config.pretrained_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OverLoCK-T evaluation")
    print(f"  CUDA device: {torch.cuda.get_device_name()}")

    if config.pretrained_path and not os.path.exists(config.pretrained_path):
        print(f"  Warning: pretrained file not found: {config.pretrained_path}")

    return config


def create_config_from_args(args) -> OverLoCKConfiguration:
    """Create config from CLI args."""
    config = OverLoCKConfiguration()

    config.backbone = args.backbone
    config.fusion_mode = args.fusion_mode
    config.dataset = args.dataset
    config.data_folder = args.data_folder
    config.img_size = args.img_size
    config.batch_size_eval = args.batch_size_eval
    config.eval_gallery_n = args.eval_gallery_n
    config.drone_height = args.drone_height
    config.train_split = args.train_split
    config.split_seed = args.split_seed
    config.attention = args.attention_type
    config.aggregation = args.aggregation_type

    if args.num_workers is not None:
        config.num_workers = args.num_workers

    return config


def split_dataset_ids(data_folder, train_split=0.7, seed=42):
    """Split dataset IDs into train/test sets."""
    satellite_folder = os.path.join(data_folder, "satellite-view")
    drone_folder = os.path.join(data_folder, "drone_view_512")

    sat_ids = set(os.listdir(satellite_folder))
    drone_ids = set(os.listdir(drone_folder))
    common_ids = list(sat_ids.intersection(drone_ids))
    common_ids.sort()

    random.seed(seed)
    random.shuffle(common_ids)

    split_idx = int(len(common_ids) * train_split)
    train_ids = common_ids[:split_idx]
    test_ids = common_ids[split_idx:]

    print(f"Total IDs: {len(common_ids)}")
    print(f"Train IDs: {len(train_ids)}")
    print(f"Test IDs: {len(test_ids)}")

    return train_ids, test_ids


def evaluate_with_reranked_topk(reranked_indices, img_features_query, img_features_gallery, ql, gl, ranks):
    """Evaluate using reranked Top-K indices."""
    num_query = len(ql)
    num_gallery = len(gl)

    CMC = torch.IntTensor(num_gallery).zero_()
    ap = 0.0

    for i in tqdm(range(num_query), desc="Reranked eval"):
        reranked_topk = reranked_indices[i]

        score = img_features_gallery @ img_features_query[i].unsqueeze(-1)
        score = score.squeeze().cpu().numpy()
        full_index = np.argsort(score)[::-1]

        topk_set = set(reranked_topk)
        remaining = [idx for idx in full_index if idx not in topk_set]
        final_index = np.array(reranked_topk + remaining)

        query_index = np.argwhere(gl == ql[i])
        good_index = query_index
        junk_index = np.argwhere(gl == -1)

        ap_tmp, CMC_tmp = compute_mAP(final_index, good_index, junk_index)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / num_query * 100
    CMC = CMC.float() / num_query

    return {"CMC": CMC, "mAP": AP}


def rerank_wrong_queries_only(reranker, top_k_indices, top_k_scores, wrong_query_indices, num_total, desc):
    """Rerank only the queries with wrong top-1 (affine Sampson)."""
    reranked_indices = [top_k_indices[i].cpu().tolist() for i in range(num_total)]

    for q_idx in tqdm(wrong_query_indices, desc=desc):
        candidates = top_k_indices[q_idx].cpu().tolist()
        initial_scores = top_k_scores[q_idx].cpu().numpy()
        reranked = reranker.rerank_topk(
            query_idx=q_idx,
            candidate_indices=candidates,
            initial_scores=initial_scores,
        )
        reranked_indices[q_idx] = reranked

    return reranked_indices


def rerank_wrong_queries_only_magsac(reranker, top_k_indices, top_k_scores, wrong_query_indices, num_total, desc):
    """Rerank only the queries with wrong top-1 (MAGSAC++)."""
    reranked_indices = [top_k_indices[i].cpu().tolist() for i in range(num_total)]

    for q_idx in tqdm(wrong_query_indices, desc=desc):
        candidates = top_k_indices[q_idx].cpu().tolist()
        initial_scores = top_k_scores[q_idx].cpu().numpy()
        reranked = reranker.rerank_topk_magsac(
            query_idx=q_idx,
            candidate_indices=candidates,
            initial_scores=initial_scores,
        )
        reranked_indices[q_idx] = reranked

    return reranked_indices


def format_eval_string(results, ranks, top1):
    parts = []
    for r in ranks:
        parts.append(f"Recall@{r}: {results['CMC'][r - 1] * 100:.4f}")
    parts.append(f"Recall@top1: {results['CMC'][top1] * 100:.4f}")
    parts.append(f"AP: {results['mAP']:.4f}")
    return " - ".join(parts)


def print_comparison_results_simple(r1_baseline, sparse, magsac, semidense, semidense_magsac, num_wrong, num_total):
    """Print simple comparison summary."""
    baseline_r1 = r1_baseline.item() if hasattr(r1_baseline, "item") else r1_baseline
    sparse_r1 = sparse["CMC"][0].item()
    magsac_r1 = magsac["CMC"][0].item()
    semidense_r1 = semidense["CMC"][0].item()
    semidense_magsac_r1 = semidense_magsac["CMC"][0].item()

    print("\n=== Summary ===")
    if num_wrong is not None and num_total is not None:
        print(f"Rerank targets: {num_wrong}/{num_total} queries")

    print(f"1) Baseline R@1: {baseline_r1 * 100:.4f} (AP: see baseline output)")
    print(f"2) Sparse + affine R@1: {sparse_r1 * 100:.4f} (AP: {sparse['mAP']:.4f})")
    print(f"3) Sparse + MAGSAC++ R@1: {magsac_r1 * 100:.4f} (AP: {magsac['mAP']:.4f})")
    print(f"4) Semi-dense + affine R@1: {semidense_r1 * 100:.4f} (AP: {semidense['mAP']:.4f})")
    print(f"5) Semi-dense + MAGSAC++ R@1: {semidense_magsac_r1 * 100:.4f} (AP: {semidense_magsac['mAP']:.4f})")


def evaluate_all_modes(
    config,
    model,
    query_loader,
    gallery_loader,
    ranks=(1, 5, 10),
    rerank_top_k=10,
    save_visualization=False,
    vis_output_dir="./visualization",
    vis_top_k=3,
):
    """Evaluate baseline and all geometric re-ranking modes."""
    from utils.geometric_reranker import GeometricReranker

    print("\n=== Eval 1: baseline (no rerank) ===")
    r1_baseline, details = evaluate(
        config=config,
        model=model,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        ranks=list(ranks),
        step_size=1000,
        cleanup=False,
        return_details=True,
    )

    img_features_query = details["img_features_query"]
    img_features_gallery = details["img_features_gallery"]
    wrong_query_indices = details["wrong_query_indices"]
    ql = details["ql"]
    gl = details["gl"]

    num_total = len(ql)
    num_gallery = len(gl)
    num_wrong = len(wrong_query_indices)

    similarity_matrix = img_features_query @ img_features_gallery.T
    top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=rerank_top_k, dim=1)

    top1 = round(num_gallery * 0.01)
    print(f"Top-1 correct: {num_total - num_wrong}/{num_total}")
    print(f"Top-1 wrong:   {num_wrong}/{num_total}")

    print("\n=== Eval 2: XFeat sparse + affine Sampson ===")
    reranker_sparse = GeometricReranker(
        device=config.device,
        top_k_rerank=rerank_top_k,
        match_mode="sparse",
    )
    reranker_sparse.setup_image_paths(
        query_dataset=query_loader.dataset,
        gallery_dataset=gallery_loader.dataset,
    )
    reranked_sparse = rerank_wrong_queries_only(
        reranker_sparse,
        top_k_indices,
        top_k_scores,
        wrong_query_indices,
        num_total,
        desc="Sparse rerank",
    )
    results_sparse = evaluate_with_reranked_topk(
        reranked_sparse,
        img_features_query,
        img_features_gallery,
        ql,
        gl,
        ranks,
    )
    print(format_eval_string(results_sparse, ranks, top1))

    print("\n=== Eval 3: XFeat sparse + MAGSAC++ ===")
    reranked_magsac = rerank_wrong_queries_only_magsac(
        reranker_sparse,
        top_k_indices,
        top_k_scores,
        wrong_query_indices,
        num_total,
        desc="Sparse MAGSAC++ rerank",
    )
    results_magsac = evaluate_with_reranked_topk(
        reranked_magsac,
        img_features_query,
        img_features_gallery,
        ql,
        gl,
        ranks,
    )
    print(format_eval_string(results_magsac, ranks, top1))

    print("\n=== Eval 4: XFeat semi-dense + affine Sampson ===")
    reranker_semidense = GeometricReranker(
        device=config.device,
        top_k_rerank=rerank_top_k,
        match_mode="semi_dense",
    )
    reranker_semidense.setup_image_paths(
        query_dataset=query_loader.dataset,
        gallery_dataset=gallery_loader.dataset,
    )
    reranked_semidense = rerank_wrong_queries_only(
        reranker_semidense,
        top_k_indices,
        top_k_scores,
        wrong_query_indices,
        num_total,
        desc="Semi-dense rerank",
    )
    results_semidense = evaluate_with_reranked_topk(
        reranked_semidense,
        img_features_query,
        img_features_gallery,
        ql,
        gl,
        ranks,
    )
    print(format_eval_string(results_semidense, ranks, top1))

    print("\n=== Eval 5: XFeat semi-dense + MAGSAC++ ===")
    reranked_semidense_magsac = rerank_wrong_queries_only_magsac(
        reranker_semidense,
        top_k_indices,
        top_k_scores,
        wrong_query_indices,
        num_total,
        desc="Semi-dense MAGSAC++ rerank",
    )
    results_semidense_magsac = evaluate_with_reranked_topk(
        reranked_semidense_magsac,
        img_features_query,
        img_features_gallery,
        ql,
        gl,
        ranks,
    )
    print(format_eval_string(results_semidense_magsac, ranks, top1))

    print_comparison_results_simple(
        r1_baseline,
        results_sparse,
        results_magsac,
        results_semidense,
        results_semidense_magsac,
        num_wrong=num_wrong,
        num_total=num_total,
    )

    if save_visualization:
        from utils.visualization import save_all_visualizations

        save_all_visualizations(
            query_dataset=query_loader.dataset,
            gallery_dataset=gallery_loader.dataset,
            wrong_query_indices=wrong_query_indices,
            top_k_indices=top_k_indices,
            reranked_indices=reranked_magsac,
            ql=ql,
            gl=gl,
            output_dir=vis_output_dir,
            top_k=vis_top_k,
        )
        print(f"Visualization saved to: {vis_output_dir}")

    del img_features_query, img_features_gallery, similarity_matrix
    del top_k_scores, top_k_indices, details
    gc.collect()
    torch.cuda.empty_cache()

    return r1_baseline, results_sparse, results_magsac, results_semidense, results_semidense_magsac


def load_model_weights(model, model_path, device):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            cleaned[k.replace("module.", "")] = v
        state = cleaned
    model.load_state_dict(state, strict=True)


if __name__ == "__main__":
    args = parse_args()

    config = create_config_from_args(args)
    config = setup_overlock_config(config)

    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.cudnn_deterministic

    # Dataset paths
    config.satellite_folder = os.path.join(config.data_folder, "satellite-view")
    config.drone_folder = os.path.join(config.data_folder, "drone_view_512")

    _, test_ids = split_dataset_ids(config.data_folder, config.train_split, config.split_seed)

    print("\n=== Prediction setup ===")
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {config.dataset}")
    print(f"Data folder: {config.data_folder}")
    print(f"Drone height: {config.drone_height}")
    print(f"Train/Test split: {config.train_split:.1%}/{1 - config.train_split:.1%}")

    # Transforms
    val_transforms, _, _ = get_transforms((config.img_size, config.img_size))

    # Model
    print("\n=== Model init ===")
    model = GeoModelOverLoCK(config)
    print(f"Loading checkpoint: {args.model_path}")
    load_model_weights(model, args.model_path, config.device)
    model = model.to(config.device)
    model.eval()

    # Datasets
    print("\n=== Data loading ===")
    if config.dataset == "SUES200-D2S":
        query_dataset_test = SUES200DatasetEval(
            data_folder=config.drone_folder,
            mode="query",
            drone_height=config.drone_height,
            class_ids=test_ids,
            transforms=val_transforms,
        )
        gallery_dataset_test = SUES200DatasetEval(
            data_folder=config.satellite_folder,
            mode="gallery",
            drone_height=None,
            class_ids=test_ids,
            transforms=val_transforms,
            sample_ids=query_dataset_test.get_sample_ids(),
            gallery_n=config.eval_gallery_n,
        )
    else:
        query_dataset_test = SUES200DatasetEval(
            data_folder=config.satellite_folder,
            mode="query",
            drone_height=None,
            class_ids=test_ids,
            transforms=val_transforms,
        )
        gallery_dataset_test = SUES200DatasetEval(
            data_folder=config.drone_folder,
            mode="gallery",
            drone_height=config.drone_height,
            class_ids=test_ids,
            transforms=val_transforms,
            sample_ids=query_dataset_test.get_sample_ids(),
            gallery_n=config.eval_gallery_n,
        )

    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    gallery_dataloader_test = DataLoader(
        gallery_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Query size: {len(query_dataset_test)}")
    print(f"Gallery size: {len(gallery_dataset_test)}")

    if args.enable_geometric_rerank:
        print(f"\n=== Evaluation with geometric rerank (Top-{args.geometric_rerank_top_k}) ===")
        r1_baseline, results_sparse, results_magsac, results_semidense, results_semidense_magsac = evaluate_all_modes(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test,
            ranks=(1, 5, 10),
            rerank_top_k=args.geometric_rerank_top_k,
            save_visualization=args.save_visualization,
            vis_output_dir=args.vis_output_dir,
            vis_top_k=args.vis_top_k,
        )

        r1_baseline_val = r1_baseline.item() if hasattr(r1_baseline, "item") else r1_baseline
        print("\n=== Final summary ===")
        print(f"Baseline R@1: {r1_baseline_val * 100:.4f}")
        print(f"Sparse + affine R@1: {results_sparse['CMC'][0].item() * 100:.4f}")
        print(f"Sparse + MAGSAC++ R@1: {results_magsac['CMC'][0].item() * 100:.4f}")
        print(f"Semi-dense + affine R@1: {results_semidense['CMC'][0].item() * 100:.4f}")
        print(f"Semi-dense + MAGSAC++ R@1: {results_semidense_magsac['CMC'][0].item() * 100:.4f}")
    else:
        print("\n=== Evaluation (no rerank) ===")
        r1_test = evaluate(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test,
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True,
        )
        print(f"\nRecall@1: {r1_test:.4f}")
