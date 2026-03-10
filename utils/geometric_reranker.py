"""
几何重排序模块 (XFeat + 仿射桑普森)

模块依赖:
- xfeat/modules/xfeat.py
- S3Esti/abso_esti_model/abso_esti_net.py (EstiNet)
- LearningACs/Network/essential_matrix_affine_without_kornia.py

关键辅助函数:
- laf_from_ours: 从坐标/尺度/角度构建 LAF
- get_laf_orientation: 获取 LAF 方向
- set_laf_orientation: 设置/恢复 LAF 方向
- scale_laf: 调整 LAF 尺度
- get_affine_correspondences: 从 LAF 对提取仿射对应
"""

import sys
import os
import cv2
import torch
import numpy as np
from typing import List, Union

# 添加模块路径
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'xfeat'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'LearningACs'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'S3Esti'))  # 添加 S3Esti 路径


class GeometricReranker:
    """
    几何重排序器
    
    使用 XFeat 特征匹配 + 仿射桑普森距离进行几何验证
    仅对 baseline Top-1 错误的 query 执行几何重排序
    
    """
    
    def __init__(self, device='cuda', top_k_rerank=10, 
                 xfeat_top_k=4096, min_matches=30,
                 match_mode='sparse'):
        """
        初始化几何重排序器
        
        Args:
            device: 计算设备（固定为 'cuda' 或 'cuda:0'）
            top_k_rerank: 重排序的候选数量
            xfeat_top_k: XFeat 提取的关键点数量
            min_matches: 最少匹配点数量阈值
            match_mode: XFeat 匹配模式
                - 'sparse': 使用 match_xfeat (稀疏特征，速度快)
                - 'semi_dense': 使用 match_xfeat_star (半稠密特征，精度高)
        
        Note:
            采用几何优先重排序（视觉分数仅用于几何分数相近时打破平局）。
        """
        self.device = self._normalize_device(device)
        self.top_k_rerank = top_k_rerank
        self.xfeat_top_k = xfeat_top_k
        self.min_matches = min_matches
        self.match_mode = match_mode
        
        # 图像路径缓存
        self.query_paths = None
        self.gallery_paths = None
        
        self._load_models()
        
        print(f"  设备: {self.device}")
        print(f"  重排序策略: 几何优先（视觉仅用于打破平局）")
        print(f"  XFeat 匹配模式: {match_mode} ({'match_xfeat' if match_mode == 'sparse' else 'match_xfeat_star'})")
    
    def _normalize_device(self, device):
        """
        设备归一化处理
        
        Args:
            device: 用户指定的设备字符串
        
        Returns:
            normalized_device: 归一化后的设备字符串（'cuda' 或 'cuda:0'）
        """
        if not torch.cuda.is_available():
            print("  CUDA 不可用，几何模块将使用 CPU")
            return 'cpu'
        
        # 归一化设备字符串
        device_str = str(device).lower()
        
        # 如果是 'cuda' 或 'cuda:0'，直接使用
        if device_str in ('cuda', 'cuda:0'):
            return 'cuda'  # 统一用 'cuda'，让 PyTorch 自动选择 cuda:0
        
        # 如果是其他 GPU（cuda:1, cuda:2 等），打印警告并降级到 cuda:0
        if device_str.startswith('cuda:'):
            print(f"  ⚠️ 几何模块设备警告：")
            print(f"     请求设备: {device}")
            print(f"     实际使用: cuda (cuda:0)")
            print(f"     原因: LearningACs 内部硬编码依赖 cuda:0，多卡环境下强制使用 cuda:0 避免设备不匹配")
            return 'cuda'
        
        # 其他情况（cpu 等）
        return device_str
    
    def _load_models(self):
        """
        加载所需模型:
        1. XFeat (特征匹配)
        2. EstiNet (尺度+角度)
        3. LAFAffNetShapeEstimator (仿射形状)
        """
        from modules.xfeat import XFeat
        # 使用 S3Esti 的 EstiNet 实现
        from abso_esti_model.abso_esti_net import EstiNet
        from Network.essential_matrix_affine_without_kornia import LAFAffNetShapeEstimator
        
        print("加载几何重排序模型...")
        
        # 1. XFeat
        xfeat_weights = os.path.join(_REPO_ROOT, 'xfeat', 'weights', 'xfeat.pt')
        self.xfeat = XFeat(weights=xfeat_weights, top_k=self.xfeat_top_k)
        print(f"  ✓ XFeat 加载完成")
        
        # 2. EstiNet (尺度和角度) - 使用 S3Esti 的权重
        esti_weights = os.path.join(_REPO_ROOT, 'S3Esti', 'abso_esti_model', 'S3Esti_ep30.pth')
        patch_size = 32
        scale_ratio_list = [0.5, 1, 2]
        
        self.model_scale = EstiNet(
            need_bn=True, device=self.device, out_channels=300,
            patch_size=patch_size, scale_ratio=scale_ratio_list
        ).to(self.device).eval()
        
        self.model_angle = EstiNet(
            need_bn=True, device=self.device, out_channels=360,
            patch_size=patch_size, scale_ratio=scale_ratio_list
        ).to(self.device).eval()
        
        checkpoint = torch.load(esti_weights, map_location=self.device)
        # S3Esti checkpoint 包含 'base' 键需要移除
        if 'base' in checkpoint['model_scale']:
            checkpoint['model_scale'].pop('base')
        if 'base' in checkpoint['model_angle']:
            checkpoint['model_angle'].pop('base')
        self.model_scale.load_state_dict(checkpoint['model_scale'], strict=True)
        self.model_angle.load_state_dict(checkpoint['model_angle'], strict=True)
        print(f"  ✓ EstiNet 加载完成 (使用 S3Esti 权重)")
        
        # 3. LAFAffNetShapeEstimator
        # 原 Aff_res_shape.pth 是 4 通道输出，与 LAFAffNetShapeEstimator (3通道) 不兼容
        # 改用 S3Esti/AffNet/AffNet_det.pth，它是 3 通道输出，完全兼容
        affnet_weights = os.path.join(_REPO_ROOT, 'S3Esti', 'AffNet', 'AffNet_det.pth')
        self.affnet = LAFAffNetShapeEstimator(
            pretrained=True,
            weight_path=affnet_weights
        ).to(self.device).eval()
        print(f"  ✓ LAFAffNet 加载完成 (使用 S3Esti 权重)")
    
    def setup_image_paths(self, query_dataset, gallery_dataset):
        """
        从 Dataset 中提取图像路径
        """
        self.query_paths = query_dataset.images      # List[str]
        self.gallery_paths = gallery_dataset.images  # List[str]
        print(f"  图像路径已设置: {len(self.query_paths)} queries, {len(self.gallery_paths)} galleries")
    
    def load_image_for_xfeat(self, img_path):
        """
        为 XFeat 加载原始图像
        Returns:
            img: np.ndarray [H, W, 3] uint8, RGB格式
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # uint8, [0, 255]
    
    def extract_matches(self, query_path, gallery_path):
        """
        使用 XFeat 提取匹配点对
        
        根据 match_mode 选择匹配方法:
        - 'sparse': match_xfeat (稀疏特征，速度快)
        - 'semi_dense': match_xfeat_star (半稠密特征，精度高)
        
        Args:
            query_path: 查询图像路径
            gallery_path: 候选图像路径
        
        Returns:
            mkpts1: np.ndarray [N, 2] 图像1的匹配点
            mkpts2: np.ndarray [N, 2] 图像2的匹配点
        """
        # 加载原始图像
        img1 = self.load_image_for_xfeat(query_path)
        img2 = self.load_image_for_xfeat(gallery_path)
        
        # 根据模式选择 XFeat 匹配方法
        with torch.no_grad():
            if self.match_mode == 'sparse':
                # 稀疏特征匹配 (速度快，~30ms/pair)
                # 官方接口: match_xfeat(img1, img2, top_k=None, min_cossim=-1)
                # min_cossim=0.82 过滤低质量匹配（官方默认-1不过滤）
                mkpts1, mkpts2 = self.xfeat.match_xfeat(
                    img1, img2, 
                    top_k=self.xfeat_top_k,
                    min_cossim=0.83  # 推荐阈值，过滤低置信度匹配
                )
            elif self.match_mode == 'semi_dense':
                # 半稠密特征匹配 (无亚像素精化版本)
                # 跨视角场景下禁用亚像素精化，使用更严格的质量过滤
                # 原因：亚像素精化假设局部平滑，在极端视角变化下可能适得其反
                # 参考: xfeat/modules/xfeat.py 第189-217行 (match_xfeat_star)
                
                # Step 1: 解析输入
                im1 = self.xfeat.parse_input(img1)
                im2 = self.xfeat.parse_input(img2)
                
                # Step 2: 提取半稠密特征
                out1 = self.xfeat.detectAndComputeDense(im1, top_k=self.xfeat_top_k)
                out2 = self.xfeat.detectAndComputeDense(im2, top_k=self.xfeat_top_k)
                
                # Step 3: 批量匹配 (提高 min_cossim 阈值，更严格的质量过滤)
                # 从 0.82 提高到 0.83，过滤更多低质量匹配
                idxs_list = self.xfeat.batch_match(
                    out1['descriptors'], out2['descriptors'], 
                    min_cossim=0.82  # 提高阈值：更严格的质量过滤
                )
                
                # Step 4: 直接使用原始坐标（禁用亚像素精化）
                # 不调用 refine_matches，避免跨视角场景下的精化误差
                # batch_match 返回 [(idx0, idx1), ...] 元组列表
                idx0, idx1 = idxs_list[0]  # 解包元组：idx0是图1的索引，idx1是图2的索引
                if len(idx0) > 0:
                    mkpts1 = out1['keypoints'][0][idx0].cpu().numpy()  # [N, 2]
                    mkpts2 = out2['keypoints'][0][idx1].cpu().numpy()  # [N, 2]
                else:
                    mkpts1, mkpts2 = np.array([]), np.array([])
            else:
                raise ValueError(f"未知的匹配模式: {self.match_mode}")
        
        return mkpts1, mkpts2
    
    def estimate_affine_params(self, query_path, gallery_path, mkpts1, mkpts2):
        """
        估计仿射参数
        
        Pipeline:
        1. 加载灰度图像 (用于 patch 提取)
        2. 加载 RGB 图像 (用于 LAFAffNet，转灰度)
        3. 生成 32×32 patches
        4. EstiNet → scale, angle
        5. laf_from_ours → 构建初始 LAF
        6. LAFAffNet → 优化 LAF (输入是 LAF + 整张图像)
        7. get_affine_correspondences → ACs
        
        Returns:
            tuple: (affine_params, filtered_mkpts1, filtered_mkpts2) 或 None
                - affine_params: np.ndarray [M, 4] 仿射参数 [a11, a12, a21, a22]
                - filtered_mkpts1: np.ndarray [M, 2] 过滤后的匹配点1
                - filtered_mkpts2: np.ndarray [M, 2] 过滤后的匹配点2
                - 如果有效点数不足，返回 None
        """
        from Network.essential_matrix_affine_without_kornia import (
            laf_from_ours, scale_laf, get_affine_correspondences
        )
        import kornia as K
        
        # 1. 加载灰度图像 (用于 patch 提取)
        gray1 = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
        gray2 = cv2.imread(gallery_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. 提取 patches
        patches1, patches2, valid_indices = self._extract_patches(
            gray1, gray2, mkpts1, mkpts2, patch_size=32
        )
        
        if len(valid_indices) < self.min_matches:
            return None
        
        # 更新有效匹配点
        mkpts1 = mkpts1[valid_indices]
        mkpts2 = mkpts2[valid_indices]
        
        # 3. EstiNet 估计尺度和角度 (官方第118-134行)
        with torch.no_grad():
            # 构建多尺度输入 [N, 9, 32, 32]
            patches1_tensor = self._build_multiscale_input(patches1)
            patches2_tensor = self._build_multiscale_input(patches2)
            
            # 估计尺度
            scale1 = self._estimate_scale(patches1_tensor)
            scale2 = self._estimate_scale(patches2_tensor)
            
            # 估计角度
            angle1 = self._estimate_angle(patches1_tensor)
            angle2 = self._estimate_angle(patches2_tensor)
        
        # 4. 构建初始 LAF (官方第149行: laf_from_ours)
        # 注意：laf_from_ours 内部会自行把坐标/尺度/角度转成 torch.Tensor
        # 因此这里保持为 numpy/list 即可，避免传 torch.Tensor 触发 Python 迭代/类型问题
        coord1 = np.asarray(mkpts1, dtype=np.float32)
        coord2 = np.asarray(mkpts2, dtype=np.float32)
        
        lafs1, _ = laf_from_ours(coord1, scale1, angle1, mrSize=6, with_resp=True, device=self.device)
        lafs2, _ = laf_from_ours(coord2, scale2, angle2, mrSize=6, with_resp=True, device=self.device)
        
        # 5. LAFAffNet 优化 (官方第142-156行)
        # 关键: LAFAffNet 需要整张图像，不是 patches
        with torch.no_grad():
            # 加载 RGB 图像并转换为灰度 tensor [1, 1, H, W]
            # 关键：确保输出是 [1, 3, H, W]，再 mean(dim=1) 得到 [1, 1, H, W]
            rgb1 = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
            rgb2 = cv2.cvtColor(cv2.imread(gallery_path), cv2.COLOR_BGR2RGB)
            # 与 LearningACs 中 load_torch_image 的用法保持一致：第二个参数传 False 以得到 [1,3,H,W]
            im1 = K.image_to_tensor(rgb1, False).float() / 255.
            im2 = K.image_to_tensor(rgb2, False).float() / 255.
            gray_tensor1 = im1.mean(dim=1, keepdim=True)  # [1, 1, H, W], float in [0,1]
            gray_tensor2 = im2.mean(dim=1, keepdim=True)  # [1, 1, H, W], float in [0,1]
            
            # LAFAffNet 优化
            lafs1 = self._refine_lafs_with_affnet(gray_tensor1, lafs1)
            lafs2 = self._refine_lafs_with_affnet(gray_tensor2, lafs2)
            
            # scale_laf 调整尺度 (官方第158-160行)
            # 注意: scale_laf 内部硬编码了 mrSize = 6
            # 必须与 laf_from_ours 的 mrSize 参数保持一致！
            lafs1 = scale_laf(lafs1, torch.tensor(scale1, device=self.device, dtype=torch.float32))
            lafs2 = scale_laf(lafs2, torch.tensor(scale2, device=self.device, dtype=torch.float32))
        
        # 6. 提取仿射对应 (官方第180行: get_affine_correspondences)
        lafs1_np = lafs1.squeeze(0).cpu().numpy()  # [M, 2, 3]
        lafs2_np = lafs2.squeeze(0).cpu().numpy()  # [M, 2, 3]
        tentatives = list(range(len(lafs1_np)))
        
        ACs = get_affine_correspondences(lafs1_np, lafs2_np, tentatives)
        # ACs: [M, 8] = [x1, y1, x2, y2, a11, a12, a21, a22]
        
        # 只返回仿射参数部分 [a11, a12, a21, a22]
        affine_params = ACs[:, 4:8]
        
        # 返回三元组：仿射参数 + 过滤后的匹配点（与官方 demo 保持一致）
        return affine_params, mkpts1, mkpts2
    
    def _extract_patches(self, gray1, gray2, mkpts1, mkpts2, patch_size=32):
        """从灰度图中提取 patches"""
        half = patch_size // 2
        patches1, patches2 = [], []
        valid_indices = []
        
        for i, (pt1, pt2) in enumerate(zip(mkpts1, mkpts2)):
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            
            # 边界检查
            if (x1-half < 0 or x1+half > gray1.shape[1] or
                y1-half < 0 or y1+half > gray1.shape[0] or
                x2-half < 0 or x2+half > gray2.shape[1] or
                y2-half < 0 or y2+half > gray2.shape[0]):
                continue
            
            p1 = gray1[y1-half:y1+half, x1-half:x1+half]
            p2 = gray2[y2-half:y2+half, x2-half:x2+half]
            
            if p1.shape == (patch_size, patch_size) and p2.shape == (patch_size, patch_size):
                patches1.append(p1)
                patches2.append(p2)
                valid_indices.append(i)
        
        if len(patches1) == 0:
            return np.array([]), np.array([]), []
        
        return np.stack(patches1), np.stack(patches2), np.array(valid_indices)
    
    def _build_multiscale_input(self, patches):
        """
        为 EstiNet 构建多尺度输入
        输入: [N, 32, 32] uint8
        输出: torch.Tensor [N, 9, 32, 32]
        """
        patches_tensor = torch.from_numpy(patches).float().to(self.device)
        patches_tensor = patches_tensor.unsqueeze(1)  # [N, 1, 32, 32]
        patches_tensor = patches_tensor.expand(-1, 9, -1, -1)  # [N, 9, 32, 32]
        return patches_tensor
    
    def _estimate_scale(self, patches_tensor):
        """
        使用 EstiNet 估计尺度
        """
        scale_resp = self.model_scale(patches_tensor)
        scale_idx = torch.argmax(scale_resp, dim=1).cpu().numpy()
        scales = self.model_scale.map_id_to_scale(scale_idx)
        return scales.cpu().numpy()
    
    def _estimate_angle(self, patches_tensor):
        """
        使用 EstiNet 估计角度
        """
        angle_resp = self.model_angle(patches_tensor)
        angle_idx = torch.argmax(angle_resp, dim=1).cpu().numpy()
        angles = self.model_angle.map_id_to_angle(angle_idx)
        return angles.cpu().numpy()
    
    def _refine_lafs_with_affnet(self, gray_img_tensor, init_lafs_tensor):
        """
        使用 LAFAffNet 优化 LAF 形状
        调用方式:
            lafs1 = affnet(lafs1.cuda(), im1.mean(dim=1, keepdim=True).cuda())
        
        Args:
            gray_img_tensor: 灰度图像 torch.Tensor [1, 1, H, W]
            init_lafs_tensor: 初始 LAF torch.Tensor [1, N, 2, 3]
        
        Returns:
            refined_lafs: 优化后的 LAF torch.Tensor [1, N, 2, 3]
        """
        from Network.essential_matrix_affine_without_kornia import (
            get_laf_orientation, set_laf_orientation
        )
        
        if init_lafs_tensor.shape[1] == 0:
            return init_lafs_tensor
        
        # 1. 保存原始方向 (官方第150行)
        ori_orig = get_laf_orientation(init_lafs_tensor)
        
        # 2. LAFAffNet 优化 (官方第151行)
        # 输入是 LAF 和整张灰度图像，不是 patches！
        refined_lafs = self.affnet(
            init_lafs_tensor.to(self.device), 
            gray_img_tensor.to(self.device)
        )
        
        # 3. 恢复原始方向 (官方第152行)
        refined_lafs = set_laf_orientation(refined_lafs, ori_orig)
        
        return refined_lafs
    
    def compute_geometric_score(self, pts1, pts2, affine_params):
        """
        计算仿射桑普森距离作为几何一致性分数
        
        Args:
            pts1: np.ndarray [N, 2] 图像1的匹配点坐标
            pts2: np.ndarray [N, 2] 图像2的匹配点坐标
            affine_params: np.ndarray [N, 4] 仿射参数 [a11, a12, a21, a22]
        
        Returns:
            score: float, 越小表示几何一致性越好
        """
        # 不需要显式拼接齐次坐标；保持 Nx2 点坐标即可
        return self._affine_sampson_distance(pts1, pts2, affine_params, homos=False)
    
    def compute_geometric_score_magsac(self, pts1, pts2):
        """
        使用 MAGSAC++ 计算几何一致性
        
        Args:
            pts1: np.ndarray [N, 2] 图像1的匹配点坐标
            pts2: np.ndarray [N, 2] 图像2的匹配点坐标
        
        Returns:
            score: float, 越小表示几何一致性越好
        """
        pts1 = np.asarray(pts1, dtype=np.float32)
        pts2 = np.asarray(pts2, dtype=np.float32)
        
        if len(pts1) < 4:
            return float('inf')
        
        # 使用 USAC_MAGSAC
        H, mask = cv2.findHomography(
            pts1, pts2,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.9999
        )
        
        if H is None or mask is None:
            return float('inf')
        
        inliers = mask.ravel().astype(bool)
        n_inliers = inliers.sum()
        inlier_ratio = n_inliers / len(inliers)
        
        # 内点太少：不可靠
        if n_inliers < 8:
            return float('inf')
        
        # 计算内点的重投影误差
        pts1_h = np.hstack([pts1[inliers], np.ones((n_inliers, 1))])
        pts2_proj = (H @ pts1_h.T).T
        pts2_proj = pts2_proj[:, :2] / (pts2_proj[:, 2:3] + 1e-8)
        reproj_error = np.linalg.norm(pts2_proj - pts2[inliers], axis=1).mean()
        
        # 组合评分（越小越好）
        # 内点比例最重要，重投影误差辅助区分
        score = (1.0 - inlier_ratio) * 100 + reproj_error
        
        return score
    
    def rerank_topk_magsac(self, query_idx: int, candidate_indices: List[int], 
                           initial_scores: Union[np.ndarray, torch.Tensor]) -> List[int]:
        """
        使用 MAGSAC++ 对单个查询的 Top-K 候选进行几何重排序
        
        Args:
            query_idx: 查询图像索引
            candidate_indices: Top-K 候选的 gallery 索引
            initial_scores: 初始相似度分数
        
        Returns:
            reranked_indices: 重排序后的索引列表
        """
        query_path = self.query_paths[query_idx]
        
        geometric_scores = []
        for gallery_idx in candidate_indices:
            gallery_path = self.gallery_paths[gallery_idx]
            
            try:
                # Step 1: XFeat 匹配
                mkpts1, mkpts2 = self.extract_matches(query_path, gallery_path)
                
                # 早停: 匹配点太少
                if len(mkpts1) < self.min_matches:
                    geometric_scores.append(float('inf'))
                    continue
                
                # Step 2: 直接使用 MAGSAC++ 评分（无需仿射参数）
                score = self.compute_geometric_score_magsac(mkpts1, mkpts2)
                geometric_scores.append(score)
                
            except Exception as e:
                print(f"  MAGSAC验证失败 ({query_idx} vs {gallery_idx}): {e}")
                geometric_scores.append(float('inf'))
        
        # 几何优先重排序
        reranked_indices = self._geometry_first_rerank(
            candidate_indices, 
            initial_scores, 
            geometric_scores
        )
        
        return reranked_indices
    
    def _affine_sampson_distance(self, pts1, pts2, affine, homos=False):
        """
        仿射桑普森距离计算
        
        数学公式:
        - dist1 = M0² / (M1² + M2² + M3² + M4² + M5² + M6²)
        - dist2 = N0² / (N1² + N2² + N3² + N4² + N5² + N6²)
        - 总距离 = mean(dist1 + dist2)
        """
        # OpenCV 的 findFundamentalMat 期望输入为 Nx2 点坐标；这里保持 (x,y) 即可
        # 注：LearningACs 中存在 homos=True 的分支，但其后续实现与 Nx3 不一致，容易触发错误。
        pts1 = np.asarray(pts1, dtype=np.float32)
        pts2 = np.asarray(pts2, dtype=np.float32)
        affine = np.asarray(affine, dtype=np.float32)
        
        # 估计基础矩阵 F
        method = cv2.USAC_ACCURATE if hasattr(cv2, "USAC_ACCURATE") else cv2.FM_RANSAC
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=method,
            ransacReprojThreshold=0.999,
            confidence=0.999
        )
        
        if F is None or F.shape != (3, 3):
            return float('inf')  # 估计失败

        # 可选：只用内点计算几何分数，更稳健
        if mask is not None:
            inlier = mask.ravel().astype(bool)
            if inlier.sum() >= 8:
                pts1 = pts1[inlier]
                pts2 = pts2[inlier]
                affine = affine[inlier]
            else:
                return float('inf')
        
        f11, f12, f13, f21, f22, f23, f31, f32, f33 = F.reshape(9)
        
        dist_list = []
        for i in range(pts1.shape[0]):
            x1, y1 = pts1[i, :2]
            x2, y2 = pts2[i, :2]
            a11, a12, a21, a22 = affine[i].reshape(4)
            
            # M 系列参数 (使用 a11, a21)
            M0 = (x1 * (a11*f11 + a21*f21) + 
                  y1 * (a11*f12 + a21*f22) + 
                  a11*f13 + a21*f23 + f11*x2 + f21*y2 + f31)
            M1 = f13 + f11*x2 + f12*y1
            M2 = a11*f12 + a21*f22
            M3 = f11
            M4 = f23 + f21*x1 + f22*y1
            M5 = a11*f11 + a21*f21
            M6 = f21
            
            # N 系列参数 (使用 a12, a22)
            N0 = (x1 * (a12*f11 + a22*f21) + 
                  y1 * (a12*f12 + a22*f22) + 
                  a12*f13 + a22*f23 + f12*x2 + f22*y2 + f32)
            N1 = f13 + f11*x1 + f12*y1
            N2 = a12*f11 + a22*f21
            N3 = f12
            N4 = f23 + f21*x1 + f22*y1
            N5 = a12*f12 + a22*f22
            N6 = f22
            
            # 计算距离
            eps = 1e-8
            dist1 = M0**2 / (M1**2 + M2**2 + M3**2 + M4**2 + M5**2 + M6**2 + eps)
            dist2 = N0**2 / (N1**2 + N2**2 + N3**2 + N4**2 + N5**2 + N6**2 + eps)
            
            dist_list.append(dist1 + dist2)
        
        return np.mean(dist_list)
    
    def rerank_topk(self, query_idx: int, candidate_indices: List[int], 
                    initial_scores: Union[np.ndarray, torch.Tensor]) -> List[int]:
        """
        对单个查询的 Top-K 候选进行几何重排序
        
        Args:
            query_idx: 查询图像索引
            candidate_indices: Top-K 候选的 gallery 索引
            initial_scores: 初始相似度分数
        
        Returns:
            reranked_indices: 重排序后的索引列表
        """
        query_path = self.query_paths[query_idx]
        
        geometric_scores = []
        for gallery_idx in candidate_indices:
            gallery_path = self.gallery_paths[gallery_idx]
            
            try:
                # Step 1: XFeat 匹配
                mkpts1, mkpts2 = self.extract_matches(query_path, gallery_path)
                
                # 早停: 匹配点太少
                if len(mkpts1) < self.min_matches:
                    geometric_scores.append(float('inf'))
                    continue
                
                # Step 2-5: 仿射参数估计和桑普森距离
                result = self.estimate_affine_params(query_path, gallery_path, mkpts1, mkpts2)
                if result is None:
                    geometric_scores.append(float('inf'))
                    continue
                
                affine_params, mkpts1_valid, mkpts2_valid = result
                score = self.compute_geometric_score(mkpts1_valid, mkpts2_valid, affine_params)
                geometric_scores.append(score)
                
            except Exception as e:
                print(f"  几何验证失败 ({query_idx} vs {gallery_idx}): {e}")
                geometric_scores.append(float('inf'))
        
        # 几何优先重排序
        reranked_indices = self._geometry_first_rerank(
            candidate_indices, 
            initial_scores, 
            geometric_scores
        )
        
        return reranked_indices
    
    def _geometry_first_rerank(self, candidate_indices, visual_scores, geo_scores):
        """
        几何优先重排序策略
        
        1. 重排序仅针对 Top-1 错误的查询，视觉排序已经"失败"
        2. 完全信任几何分数，不再依赖视觉排名
        3. 视觉分数仅用于 geo_score 相近时打破平局
        
        Args:
            candidate_indices: Top-K 候选索引 List[int]
            visual_scores: 视觉相似度分数（越大越好）np.ndarray or torch.Tensor
            geo_scores: 几何分数（越小越好，inf 表示失败）List[float]
        
        Returns:
            reranked_indices: 重排序后的索引列表
        
        排序规则:
            1. 主排序键: geo_score（升序，越小越好）
            2. 次排序键: visual_score（降序，用于打破平局）
        """
        candidate_indices = list(candidate_indices)
        
        # 处理 torch.Tensor
        if hasattr(visual_scores, 'cpu'):
            visual_scores = visual_scores.cpu().numpy()
        visual_scores = np.asarray(visual_scores, dtype=np.float32)
        geo_scores = np.asarray(geo_scores, dtype=np.float32)
        
        # 失败回退：如果全部几何分数为 inf，保持原顺序
        if np.isfinite(geo_scores).sum() == 0:
            return candidate_indices
        
        # 几何优先排序：(geo_score 升序, visual_score 降序)
        # 使用 -visual_scores 实现降序
        combined = list(zip(candidate_indices, geo_scores, -visual_scores))
        combined.sort(key=lambda x: (x[1], x[2]))  # 先按 geo 升序，再按 -visual 升序
        
        return [c[0] for c in combined]
