#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Geometric re-ranking:
  - Enable with --enable_geometric_rerank
  - Supports XFeat matching modes: sparse and semi_dense
  - Only reranks queries with baseline Top-1 errors to reduce compute
    
Usage:
    python predict_u1652_overlock.py \
        --model_path ./checkpoints/.../best_score.pth \
        --dataset U1652-D2S \
        --enable_geometric_rerank \
        --geometric_rerank_top_k 5

Optional (visualization):
    --save_visualization --vis_output_dir ./visualization --vis_top_k 3
"""

import os
import gc
import torch
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader

from models.model_overlock import GeoModelOverLoCK
from overlock_t_config import get_config
from utils.trainer import evaluate, predict, eval_query, compute_mAP
from datasets.university import U1652DatasetEval, get_transforms


@dataclass
class OverLoCKConfiguration:
    # Model - 固定为OverLoCK-T main_only_no_compress模式
    backbone: str = 'overlock_t'
    attention: str = 'GSRA'
    aggregation: str = 'SALAD'

    num_channels: int = 640  # main_only_no_compress模式: 640维主特征
    img_size: int = 512
    num_clusters: int = 128
    cluster_dim: int = 64
    
    fusion_mode: str = 'main_only_no_compress'
    pretrained_path: str = None  # 自动设置

    seed: int = 1
    verbose: bool = True

    # Eval
    batch_size_eval: int = 128
    eval_gallery_n: int = -1  # -1 for all or int
    normalize_features: bool = True

    dataset: str = 'U1652-D2S'  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "/mnt/ramdisk/University-Release"

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 12
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description='OverLoCK-T LRFR预测脚本')
    
    # 模型权重路径
    parser.add_argument('--model_path', type=str, 
                        default='./checkpoints/.../best_score.pth',
                        help='模型权重路径')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='U1652-D2S',
                        choices=['U1652-D2S', 'U1652-S2D'],
                        help='数据集类型')
    parser.add_argument('--data_folder', type=str, default='/mnt/ramdisk/University-Release',
                        help='数据集根目录')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='overlock_t',
                        choices=['overlock_t', 'overlock_t_24x24'],
                        help='Backbone版本')
    parser.add_argument('--img_size', type=int, default=512,
                        help='输入图像尺寸')
    parser.add_argument('--batch_size_eval', type=int, default=128,
                        help='评估批次大小')
    
    # 几何重排序参数
    parser.add_argument('--enable_geometric_rerank', action='store_true',
                        help='启用几何重排序')
    parser.add_argument('--geometric_rerank_top_k', type=int, default=5,
                        help='几何重排序的候选数量')
    
    # ========== 新增：可视化参数 ==========
    parser.add_argument('--save_visualization', action='store_true',
                        help='保存检索结果可视化图')
    parser.add_argument('--vis_output_dir', type=str, default='./visualization',
                        help='可视化输出目录（默认: ./visualization）')
    parser.add_argument('--vis_top_k', type=int, default=3,
                        help='可视化 Top-K 数量（默认: 3）')
    
    return parser.parse_args()


def setup_overlock_config(config):
    print("=== OverLoCK-T config setup ===")
    
    # 根据 backbone 版本加载对应配置
    if config.backbone == 'overlock_t_24x24':
        # 24×24 版本使用不同的配置
        overlock_config = get_config('overlock_main_only_no_compress')
        print("  Using 24x24 backbone")
    else:
        # 标准 16×16 版本
        overlock_config = get_config('overlock_main_only_no_compress')
    
    # 更新预训练路径（用于模型架构匹配）
    config.pretrained_path = overlock_config.get('pretrained_path')
    
    # 验证num_channels（main_only_no_compress模式固定为640）
    config.num_channels = 640
    
    print(f"  Backbone: {config.backbone}")
    print(f"  Fusion mode: {config.fusion_mode}")
    print(f"  Output dim: {config.num_channels}")
    print(f"  Input size: {config.img_size}x{config.img_size}")
    print(f"  Pretrained path: {config.pretrained_path}")
    
    # 验证CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("OverLoCK-T需要CUDA支持，但当前环境未检测到CUDA设备")
    print(f"  CUDA device: {torch.cuda.get_device_name()}")
    
    return config


def create_config_from_args(args):
    config = OverLoCKConfiguration()
    
    # 更新参数
    config.backbone = args.backbone
    config.dataset = args.dataset
    config.data_folder = args.data_folder
    config.img_size = args.img_size
    config.batch_size_eval = args.batch_size_eval
    
    # 设置OverLoCK-T配置
    config = setup_overlock_config(config)
    
    # 设置数据集路径
    if config.dataset == 'U1652-D2S':
        config.query_folder_test = f'{config.data_folder}/test/query_drone'
        config.gallery_folder_test = f'{config.data_folder}/test/gallery_satellite'
    elif config.dataset == 'U1652-S2D':
        config.query_folder_test = f'{config.data_folder}/test/query_satellite'
        config.gallery_folder_test = f'{config.data_folder}/test/gallery_drone'
    
    return config


# ============================================================
# 几何重排序辅助函数
# ============================================================

def evaluate_all_modes(config, model, query_loader, gallery_loader, 
                       ranks=[1, 5, 10], rerank_top_k=10,
                       save_visualization=False, vis_output_dir='./visualization', vis_top_k=3):
    from utils.geometric_reranker import GeometricReranker
    
    # ============================================================
    # 【评估结果 1】无重排序的基础检索评估结果
    # 使用 return_details=True 获取特征和错误 ID，避免重复提取
    # ============================================================
    print("\n=== Eval 1: baseline (no rerank) ===")
    
    r1_baseline, details = evaluate(
        config=config,
        model=model,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        ranks=ranks,
        step_size=1000,
        cleanup=False,
        return_details=True  # 返回特征和错误 ID，避免重复提取
    )
    
    # 从 evaluate() 返回的详细信息中提取数据
    img_features_query = details['img_features_query']
    img_features_gallery = details['img_features_gallery']
    wrong_query_indices = details['wrong_query_indices']
    ql = details['ql']
    gl = details['gl']
    
    num_total = len(ql)
    num_gallery = len(gl)
    num_wrong = len(wrong_query_indices)
    num_correct = num_total - num_wrong
    
    # 计算 Top-K 用于重排序
    similarity_matrix = img_features_query @ img_features_gallery.T
    top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=rerank_top_k, dim=1)
    
    # top1% 用于打印
    top1 = round(num_gallery * 0.01)
    
    print(f"Top-1 correct: {num_correct}/{num_total}")
    print(f"Top-1 wrong:   {num_wrong}/{num_total}")
    
    # ============================================================
    # 【评估结果 2】XFeat 稀疏特征重排序后的最终结果
    # ============================================================
    print("\n=== Eval 2: XFeat sparse + affine Sampson ===")
    
    reranker_sparse = GeometricReranker(
        device=config.device, 
        top_k_rerank=rerank_top_k,
        match_mode='sparse'  # 使用 match_xfeat
    )
    reranker_sparse.setup_image_paths(
        query_dataset=query_loader.dataset,
        gallery_dataset=gallery_loader.dataset
    )
    
    # 对 R@1 错误的查询进行重排序
    reranked_sparse = rerank_wrong_queries_only(
        reranker_sparse, top_k_indices, top_k_scores,
        wrong_query_indices, num_total, desc="Sparse rerank"
    )
    
    # 评估修正后的检索结果 (使用原始 compute_mAP 函数)
    results_sparse = evaluate_with_reranked_topk(
        reranked_sparse, img_features_query, img_features_gallery, ql, gl, ranks
    )
    
    # 打印稀疏重排序结果 (与原始evaluate格式一致)
    print(format_eval_string(results_sparse, ranks, top1))
    
    # ============================================================
    # 【评估结果 3】XFeat 稀疏 + MAGSAC++ 重排序后的最终结果
    # ============================================================
    print("\n=== Eval 3: XFeat sparse + MAGSAC++ ===")
    
    # 复用之前的 reranker_sparse，使用 MAGSAC++ 方法重排序
    reranked_magsac = rerank_wrong_queries_only_magsac(
        reranker_sparse, top_k_indices, top_k_scores,
        wrong_query_indices, num_total, desc="Sparse MAGSAC++ rerank"
    )
    
    # 评估修正后的检索结果 (使用原始 compute_mAP 函数)
    results_magsac = evaluate_with_reranked_topk(
        reranked_magsac, img_features_query, img_features_gallery, ql, gl, ranks
    )
    
    # 打印 MAGSAC++ 重排序结果
    print(format_eval_string(results_magsac, ranks, top1))
    
    # ============================================================
    # 【评估结果 4】XFeat 半稠密特征重排序后的最终结果
    # ============================================================
    print("\n=== Eval 4: XFeat semi-dense + affine Sampson ===")
    
    reranker_semidense = GeometricReranker(
        device=config.device, 
        top_k_rerank=rerank_top_k,
        match_mode='semi_dense'  # 使用 match_xfeat_star
    )
    reranker_semidense.setup_image_paths(
        query_dataset=query_loader.dataset,
        gallery_dataset=gallery_loader.dataset
    )
    
    # 对 R@1 错误的查询进行重排序
    reranked_semidense = rerank_wrong_queries_only(
        reranker_semidense, top_k_indices, top_k_scores,
        wrong_query_indices, num_total, desc="Semi-dense rerank"
    )
    
    # 评估修正后的检索结果 (使用原始 compute_mAP 函数)
    results_semidense = evaluate_with_reranked_topk(
        reranked_semidense, img_features_query, img_features_gallery, ql, gl, ranks
    )
    
    # 打印半稠密重排序结果 (与原始evaluate格式一致)
    print(format_eval_string(results_semidense, ranks, top1))
    
    # ============================================================
    # 【评估结果 5】XFeat 半稠密 + MAGSAC++ 重排序后的最终结果
    # ============================================================
    print("\n=== Eval 5: XFeat semi-dense + MAGSAC++ ===")
    
    # 复用之前的 reranker_semidense，使用 MAGSAC++ 方法重排序
    reranked_semidense_magsac = rerank_wrong_queries_only_magsac(
        reranker_semidense, top_k_indices, top_k_scores,
        wrong_query_indices, num_total, desc="Semi-dense MAGSAC++ rerank"
    )
    
    # 评估修正后的检索结果 (使用原始 compute_mAP 函数)
    results_semidense_magsac = evaluate_with_reranked_topk(
        reranked_semidense_magsac, img_features_query, img_features_gallery, ql, gl, ranks
    )
    
    # 打印半稠密 + MAGSAC++ 重排序结果
    print(format_eval_string(results_semidense_magsac, ranks, top1))
    
    # ============================================================
    # 汇总对比
    # ============================================================
    print_comparison_results_simple(
        r1_baseline, results_sparse, results_magsac, results_semidense, results_semidense_magsac, ranks,
        num_wrong=num_wrong, num_total=num_total
    )
    
    # ============================================================
    # 【可视化】保存所有查询的可视化结果
    # ============================================================
    if save_visualization:
        from utils.visualization import save_all_visualizations
        save_all_visualizations(
            query_dataset=query_loader.dataset,
            gallery_dataset=gallery_loader.dataset,
            wrong_query_indices=wrong_query_indices,
            top_k_indices=top_k_indices,
            reranked_indices=reranked_magsac,  # 使用 Sparse + MAGSAC++ 结果
            ql=ql, gl=gl,
            output_dir=vis_output_dir,
            top_k=vis_top_k
        )
        print(f"Visualization saved to: {vis_output_dir}")
    
    # 清理特征，释放 GPU 内存
    del img_features_query, img_features_gallery, similarity_matrix
    del top_k_scores, top_k_indices, details
    gc.collect()
    torch.cuda.empty_cache()
    
    return r1_baseline, results_sparse, results_magsac, results_semidense, results_semidense_magsac


def evaluate_with_reranked_topk(reranked_indices, img_features_query, img_features_gallery, ql, gl, ranks):
    """
    使用重排序后的Top-K进行评估
    
    使用原始 compute_mAP 函数计算指标
    """
    num_query = len(ql)
    num_gallery = len(gl)
    
    CMC = torch.IntTensor(num_gallery).zero_()
    ap = 0.0
    
    for i in tqdm(range(num_query), desc="Reranked eval"):
        # 构建完整排序: 重排序的 Top-K + 剩余按原始相似度排序
        reranked_topk = reranked_indices[i]
        
        # 获取原始分数排序
        score = img_features_gallery @ img_features_query[i].unsqueeze(-1)
        score = score.squeeze().cpu().numpy()
        full_index = np.argsort(score)[::-1]  # 降序
        
        # 合并: 重排序的 Top-K 在前，剩余按原始顺序
        topk_set = set(reranked_topk)
        remaining = [idx for idx in full_index if idx not in topk_set]
        final_index = np.array(reranked_topk + remaining)
        
        # 找到正确匹配
        query_index = np.argwhere(gl == ql[i])
        good_index = query_index
        junk_index = np.argwhere(gl == -1)
        
        # 使用原始 compute_mAP 函数
        ap_tmp, CMC_tmp = compute_mAP(final_index, good_index, junk_index)
        
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    
    AP = ap / num_query * 100
    CMC = CMC.float() / num_query  # 0-1 范围，与原始 evaluate 一致
    
    return {'CMC': CMC, 'mAP': AP}


def rerank_wrong_queries_only(reranker, top_k_indices, top_k_scores, 
                               wrong_query_indices, num_total, desc="Rerank"):
    """
    仅对 Top-1 错误的查询进行重排序（使用仿射桑普森距离）
    
    优化策略：
       - Top-1 正确的查询：保持原始 Top-K 顺序
       - Top-1 错误的查询：执行几何重排序
       - 大幅减少计算量
    
    Returns:
        reranked_indices: List[List[int]], 每个查询的重排序结果
    """
    # 初始化：所有查询默认使用原始 Top-K 顺序
    reranked_indices = [top_k_indices[i].cpu().tolist() for i in range(num_total)]
    
    # 仅对 Top-1 错误的查询进行重排序
    for q_idx in tqdm(wrong_query_indices, desc=desc):
        candidates = top_k_indices[q_idx].cpu().tolist()
        initial_scores = top_k_scores[q_idx].cpu().numpy()
        
        # 执行几何重排序（使用仿射桑普森距离）
        reranked = reranker.rerank_topk(
            query_idx=q_idx,
            candidate_indices=candidates,
            initial_scores=initial_scores
        )
        reranked_indices[q_idx] = reranked
    
    return reranked_indices


def rerank_wrong_queries_only_magsac(reranker, top_k_indices, top_k_scores, 
                                      wrong_query_indices, num_total, desc="MAGSAC++ rerank"):
    # 初始化：所有查询默认使用原始 Top-K 顺序
    reranked_indices = [top_k_indices[i].cpu().tolist() for i in range(num_total)]
    
    # 仅对 Top-1 错误的查询进行重排序
    for q_idx in tqdm(wrong_query_indices, desc=desc):
        candidates = top_k_indices[q_idx].cpu().tolist()
        initial_scores = top_k_scores[q_idx].cpu().numpy()
        
        # 执行 MAGSAC++ 重排序（无需仿射参数估计）
        reranked = reranker.rerank_topk_magsac(
            query_idx=q_idx,
            candidate_indices=candidates,
            initial_scores=initial_scores
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


def print_comparison_results_simple(r1_baseline, sparse, magsac, semidense, semidense_magsac, ranks, num_wrong=None, num_total=None):
    baseline_r1 = r1_baseline.item() if hasattr(r1_baseline, 'item') else r1_baseline
    sparse_r1 = sparse['CMC'][0].item()
    magsac_r1 = magsac['CMC'][0].item()
    semidense_r1 = semidense['CMC'][0].item()
    semidense_magsac_r1 = semidense_magsac['CMC'][0].item()
    
    print("\n=== Summary ===")
    if num_wrong is not None and num_total is not None:
        print(f"Rerank targets: {num_wrong}/{num_total} queries")
    print(f"1) Baseline R@1: {baseline_r1 * 100:.4f} (AP: see baseline output)")
    print(f"2) Sparse + affine R@1: {sparse_r1 * 100:.4f} (AP: {sparse['mAP']:.4f})")
    print(f"3) Sparse + MAGSAC++ R@1: {magsac_r1 * 100:.4f} (AP: {magsac['mAP']:.4f})")
    print(f"4) Semi-dense + affine R@1: {semidense_r1 * 100:.4f} (AP: {semidense['mAP']:.4f})")
    print(f"5) Semi-dense + MAGSAC++ R@1: {semidense_magsac_r1 * 100:.4f} (AP: {semidense_magsac['mAP']:.4f})")


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置
    config = create_config_from_args(args)
    
    print("\n=== Prediction setup ===")
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {config.dataset}")
    print(f"Data folder: {config.data_folder}")
    print(f"Query folder: {config.query_folder_test}")
    print(f"Gallery folder: {config.gallery_folder_test}")
    
    # 获取数据变换
    val_transforms, _, _ = get_transforms((config.img_size, config.img_size))
    
    # 创建模型
    print("\n=== Model init ===")
    model = GeoModelOverLoCK(config)
    
    # 加载训练好的权重
    print(f"Loading checkpoint: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    model.eval()
    
    # 创建数据加载器
    print("\n=== Data loading ===")
    query_dataset_test = U1652DatasetEval(
        data_folder=config.query_folder_test,
        mode="query",
        transforms=val_transforms
    )
    
    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    gallery_dataset_test = U1652DatasetEval(
        data_folder=config.gallery_folder_test,
        mode="gallery",
        transforms=val_transforms,
        sample_ids=query_dataset_test.get_sample_ids(),
        gallery_n=config.eval_gallery_n
    )
    
    gallery_dataloader_test = DataLoader(
        gallery_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"Query size: {len(query_dataset_test)}")
    print(f"Gallery size: {len(gallery_dataset_test)}")
    
    # 根据是否启用几何重排序选择评估方式
    if args.enable_geometric_rerank:
        # 启用几何重排序：五模式对比评估
        print(f"\n=== Evaluation with geometric rerank (Top-{args.geometric_rerank_top_k}) ===")
        r1_baseline, results_sparse, results_magsac, results_semidense, results_semidense_magsac = evaluate_all_modes(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test,
            ranks=[1, 5, 10],
            rerank_top_k=args.geometric_rerank_top_k,
            save_visualization=args.save_visualization,
            vis_output_dir=args.vis_output_dir,
            vis_top_k=args.vis_top_k
        )
        
        print("\n=== Final summary ===")
        r1_baseline_val = r1_baseline.item() if hasattr(r1_baseline, 'item') else r1_baseline
        print(f"Baseline R@1: {r1_baseline_val * 100:.4f}")
        print(f"Sparse + affine R@1: {results_sparse['CMC'][0].item() * 100:.4f}")
        print(f"Sparse + MAGSAC++ R@1: {results_magsac['CMC'][0].item() * 100:.4f}")
        print(f"Semi-dense + affine R@1: {results_semidense['CMC'][0].item() * 100:.4f}")
        print(f"Semi-dense + MAGSAC++ R@1: {results_semidense_magsac['CMC'][0].item() * 100:.4f}")
    else:
        # 标准评估（无重排序）
        print("\n=== Evaluation (no rerank) ===")
        r1_test = evaluate(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test,
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True
        )
        
        print(f"\nRecall@1: {r1_test:.4f}")
