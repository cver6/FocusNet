"""
功能：
- 保存所有查询的可视化结果
- R@1 错误 → 拼接重排序前后对比图 (Before + After)
- R@1 正确 → 仅保存 Top-3 检索结果
- 使用 PIL 库，输出 PNG 图像
"""

from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np


def _load_font(size=20):
    """
    加载字体（优先使用系统字体，失败则使用默认字体）
    """
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    
    # 使用默认字体
    return ImageFont.load_default()


def _draw_centered_text(draw, text, center_x, top_y, font, fill=(0, 0, 0)):
    """
    绘制居中文本（支持多行）
    
    Args:
        draw: ImageDraw 对象
        text: 文本内容（可包含\n）
        center_x: 居中的x坐标
        top_y: 文本顶部y坐标
        font: 字体
        fill: 文本颜色
    """
    lines = text.split('\n')
    y_offset = top_y
    
    for line in lines:
        # 获取文本边界框
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 计算居中位置
        x = center_x - text_width // 2
        draw.text((x, y_offset), line, fill=fill, font=font)
        
        y_offset += text_height + 2  # 行间距


def _add_border(img, color, width=5):
    """
    给图像添加边框
    
    Args:
        img: PIL Image
        color: 边框颜色 (R, G, B)
        width: 边框宽度
    
    Returns:
        带边框的新图像
    """
    bordered = Image.new('RGB', (img.width + 2*width, img.height + 2*width), color)
    bordered.paste(img, (width, width))
    return bordered


def _resize_image(img, target_size=256):
    """
    等比例缩放图像到目标尺寸
    """
    ratio = target_size / max(img.width, img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)


def save_retrieval_result(
    query_path,              # str，查询图像路径
    gallery_paths,           # List[str]，Top-K 画廊路径
    query_id,                # int，查询地点 ID
    gallery_ids,             # List[int]，Top-K 画廊 ID
    output_path              # str，输出文件路径
):
    """
    保存 R@1 正确查询的检索结果
    布局：Query + Top-3
    
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │  Query  │ │  Top-1  │ │  Top-2  │ │  Top-3  │
    │ ID:1234 │ │ ID:1234 │ │ ID:5678 │ │ ID:9012 │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘
                   ✓绿框       ✗红框       ✗红框
    """
    # 配置
    img_size = 256
    border_width = 5
    padding = 10
    text_height = 30
    
    font = _load_font(18)
    
    # 加载并处理图像
    images = []
    labels = []
    borders = []
    
    # Query 图像
    query_img = Image.open(query_path).convert('RGB')
    query_img = _resize_image(query_img, img_size)
    images.append(query_img)
    labels.append(f"Query\nID: {query_id}")
    borders.append(None)
    
    # Top-K 画廊图像
    for i, (gpath, gid) in enumerate(zip(gallery_paths, gallery_ids)):
        gimg = Image.open(gpath).convert('RGB')
        gimg = _resize_image(gimg, img_size)
        
        is_correct = (gid == query_id)
        border_color = (0, 200, 0) if is_correct else (200, 0, 0)
        
        images.append(gimg)
        labels.append(f"Top-{i+1}\nID: {gid}")
        borders.append(border_color)
    
    # 计算画布尺寸
    num_images = len(images)
    canvas_width = num_images * (img_size + 2*border_width) + (num_images + 1) * padding
    canvas_height = img_size + 2*border_width + text_height + 2*padding
    
    # 创建画布
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 绘制图像和标签
    x_offset = padding
    for img, label, border in zip(images, labels, borders):
        # 添加边框
        if border is not None:
            img_with_border = _add_border(img, border, border_width)
        else:
            # Query 使用灰色边框
            img_with_border = _add_border(img, (128, 128, 128), border_width)
        
        # 粘贴图像
        canvas.paste(img_with_border, (x_offset, padding))
        
        # 绘制标签
        text_x = x_offset + (img_size + 2*border_width) // 2
        text_y = padding + img_size + 2*border_width + 5
        _draw_centered_text(draw, label, text_x, text_y, font)
        
        x_offset += img_size + 2*border_width + padding
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path, 'PNG')


def save_rerank_comparison(
    query_path,              # str，查询图像路径
    before_paths,            # List[str]，重排序前 Top-K 画廊路径
    after_paths,             # List[str]，重排序后 Top-K 画廊路径
    query_id,                # int，查询地点 ID
    before_ids,              # List[int]，重排序前 Top-K 画廊 ID
    after_ids,               # List[int]，重排序后 Top-K 画廊 ID
    output_path              # str，输出文件路径
):
    """
    保存 R@1 错误查询的重排序对比图
    
    布局：
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        Query ID: 1234                               │
    ├─────────────────────────────────────────────────────────────────────┤
    │ ┌───────┐  ┌─────┐ ┌─────┐ ┌─────┐  ┌─────┐ ┌─────┐ ┌─────┐       │
    │ │ Query │  │ ✗   │ │ ✗   │ │ ✓   │  │ ✓   │ │ ✗   │ │ ✗   │       │
    │ └───────┘  │5678 │ │9012 │ │1234 │  │1234 │ │5678 │ │9012 │       │
    │            └─────┘ └─────┘ └─────┘  └─────┘ └─────┘ └─────┘       │
    └─────────────────────────────────────────────────────────────────────┘
    """
    # 配置
    img_size = 200
    border_width = 4
    padding = 8
    text_height = 40
    title_height = 50
    section_gap = 30  # Before 和 After 之间的间隔
    
    font = _load_font(16)
    title_font = _load_font(20)
    
    # 计算尺寸
    top_k = len(before_paths)
    single_img_width = img_size + 2*border_width
    
    # Query 区域宽度
    query_width = single_img_width + 2*padding
    
    # Before 区域宽度
    before_width = top_k * single_img_width + (top_k + 1) * padding
    
    # After 区域宽度
    after_width = top_k * single_img_width + (top_k + 1) * padding
    
    # 总画布宽度
    canvas_width = query_width + before_width + after_width + section_gap * 2
    canvas_height = title_height + img_size + 2*border_width + text_height + 3*padding
    
    # 创建画布
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 绘制标题
    title_text = f"Query ID: {query_id}"
    _draw_centered_text(draw, title_text, canvas_width // 2, 10, title_font, fill=(0, 0, 0))
    
    # 绘制分隔线
    y_line = title_height - 5
    draw.line([(0, y_line), (canvas_width, y_line)], fill=(200, 200, 200), width=2)
    
    # ==================== Query 区域 ====================
    x_offset = padding
    y_offset = title_height + padding
    
    # Query 图像
    query_img = Image.open(query_path).convert('RGB')
    query_img = _resize_image(query_img, img_size)
    query_img = _add_border(query_img, (100, 100, 100), border_width)
    canvas.paste(query_img, (x_offset, y_offset))
    
    # Query 标签
    _draw_centered_text(
        draw, f"Query\nID: {query_id}",
        x_offset + single_img_width // 2, y_offset + single_img_width + 5,
        font
    )
    
    # ==================== Before 区域 ====================
    x_offset = query_width + section_gap
    
    # Before 图像
    for i, (bpath, bid) in enumerate(zip(before_paths, before_ids)):
        bimg = Image.open(bpath).convert('RGB')
        bimg = _resize_image(bimg, img_size)
        
        is_correct = (bid == query_id)
        border_color = (0, 200, 0) if is_correct else (200, 0, 0)
        bimg = _add_border(bimg, border_color, border_width)
        
        img_x = x_offset + padding + i * (single_img_width + padding)
        canvas.paste(bimg, (img_x, y_offset))
        
        # 标签
        _draw_centered_text(
            draw, f"Top-{i+1}\nID: {bid}",
            img_x + single_img_width // 2, y_offset + single_img_width + 5,
            font
        )
    
    # ==================== After 区域 ====================
    x_offset = query_width + before_width + section_gap * 2
    
    # After 图像
    for i, (apath, aid) in enumerate(zip(after_paths, after_ids)):
        aimg = Image.open(apath).convert('RGB')
        aimg = _resize_image(aimg, img_size)
        
        is_correct = (aid == query_id)
        border_color = (0, 200, 0) if is_correct else (200, 0, 0)
        aimg = _add_border(aimg, border_color, border_width)
        
        img_x = x_offset + padding + i * (single_img_width + padding)
        canvas.paste(aimg, (img_x, y_offset))
        
        # 标签
        _draw_centered_text(
            draw, f"Top-{i+1}\nID: {aid}",
            img_x + single_img_width // 2, y_offset + single_img_width + 5,
            font
        )
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path, 'PNG')


def save_all_visualizations(
    query_dataset,           # U1652DatasetEval，包含 .images 字段
    gallery_dataset,         # U1652DatasetEval，包含 .images 字段
    wrong_query_indices,     # np.array，R@1 错误的查询索引
    top_k_indices,           # torch.Tensor [num_query, k]，baseline Top-K
    reranked_indices,        # List[List[int]]，重排序后的 Top-K（仅错误查询有效）
    ql, gl,                  # np.array，查询/画廊标签
    output_dir,              # str，输出目录
    top_k=3                  # int，保存 Top-K 个结果
):
    """
    保存所有查询的可视化结果
    
    - R@1 错误 → save_rerank_comparison() 拼接 Before + After
    - R@1 正确 → save_retrieval_result() 仅保存 Top-3
    
    输出目录结构：
        visualization/
        ├── wrong/                              # R@1 错误（拼接对比图）
        │   └── q{idx:04d}_qid{id}/
        │       └── compare_before_after.png
        └── correct/                            # R@1 正确（仅 Top-3）
            └── q{idx:04d}_qid{id}/
                └── retrieval_top3.png
    """
    import torch
    from tqdm import tqdm
    
    # 获取图像路径列表
    query_images = query_dataset.images
    gallery_images = gallery_dataset.images
    
    num_query = len(ql)
    wrong_set = set(wrong_query_indices.tolist()) if isinstance(wrong_query_indices, np.ndarray) else set(wrong_query_indices)
    
    # 创建输出目录
    wrong_dir = os.path.join(output_dir, "wrong")
    correct_dir = os.path.join(output_dir, "correct")
    os.makedirs(wrong_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)
    
    # 统计
    num_wrong_saved = 0
    num_correct_saved = 0
    
    print(f"   可视化统计:")
    print(f"   总查询数: {num_query}")
    print(f"   R@1 错误: {len(wrong_set)}")
    print(f"   R@1 正确: {num_query - len(wrong_set)}")
    print(f"   保存 Top-{top_k} 结果")
    
    for q_idx in tqdm(range(num_query), desc="保存可视化结果"):
        query_id = int(ql[q_idx])
        query_path = query_images[q_idx]
        
        # 获取 baseline Top-K 索引
        if torch.is_tensor(top_k_indices):
            baseline_topk = top_k_indices[q_idx][:top_k].cpu().tolist()
        else:
            baseline_topk = list(top_k_indices[q_idx][:top_k])
        
        # 获取 Top-K 画廊路径和 ID
        baseline_paths = [gallery_images[idx] for idx in baseline_topk]
        baseline_ids = [int(gl[idx]) for idx in baseline_topk]
        
        if q_idx in wrong_set:
            # ==================== R@1 错误：保存对比图 ====================
            # 获取重排序后的 Top-K
            reranked_topk = reranked_indices[q_idx][:top_k]
            reranked_paths = [gallery_images[idx] for idx in reranked_topk]
            reranked_ids = [int(gl[idx]) for idx in reranked_topk]
            
            # 输出路径
            subdir = f"q{q_idx:04d}_qid{query_id}"
            output_path = os.path.join(wrong_dir, subdir, "compare_before_after.png")
            
            save_rerank_comparison(
                query_path=query_path,
                before_paths=baseline_paths,
                after_paths=reranked_paths,
                query_id=query_id,
                before_ids=baseline_ids,
                after_ids=reranked_ids,
                output_path=output_path
            )
            num_wrong_saved += 1
        else:
            # ==================== R@1 正确：保存 Top-3 ====================
            subdir = f"q{q_idx:04d}_qid{query_id}"
            output_path = os.path.join(correct_dir, subdir, "retrieval_top3.png")
            
            save_retrieval_result(
                query_path=query_path,
                gallery_paths=baseline_paths,
                query_id=query_id,
                gallery_ids=baseline_ids,
                output_path=output_path
            )
            num_correct_saved += 1
    
    print(f"\n 可视化保存完成:")
    print(f"   R@1 错误对比图: {num_wrong_saved} 张 → {wrong_dir}")
    print(f"   R@1 正确检索图: {num_correct_saved} 张 → {correct_dir}")
