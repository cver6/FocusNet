"""
参考论文: Cross-Batch Memory for Embedding Learning (CVPR 2020 Oral)
参考实现: https://github.com/msight-tech/research-xbm

核心思想:
- 维护一个FIFO队列存储历史批次的特征和标签
- 在计算损失时，将当前批次特征与队列中的历史特征一起使用
- 大幅扩展负样本池，从batch_size扩展到memory_size
"""

import torch

class XBM:
    """
    Cross-Batch Memory
    
    参考research-xbm代码实现
    
    Args:
        memory_size: 队列大小，存储多少个特征
        feat_dim: 特征维度
        device: 设备 ('cuda' or 'cpu')
    
    Attributes:
        feats1: 存储特征1（卫星图像特征）的队列 [memory_size, feat_dim]
        feats2: 存储特征2（无人机图像特征）的队列 [memory_size, feat_dim]
        ptr: 当前队列指针位置
        is_full: 队列是否已满
    """
    
    def __init__(self, memory_size, feat_dim, device='cuda'):
        self.memory_size = memory_size
        self.feat_dim = feat_dim
        self.device = device
        
        self.feats1 = torch.zeros(memory_size, feat_dim, device=device)
        self.feats2 = torch.zeros(memory_size, feat_dim, device=device)
        # 存储labels（用于区分正负样本）
        self.labels = torch.zeros(memory_size, dtype=torch.long, device=device)
        
        # 队列指针，指向下一个要插入的位置
        self.ptr = 0
        # 使用显式布尔标志判断队列是否已满
        # 避免依赖特征值判断，防止特征坍塌/半精度/零向量等场景下的误判
        self.has_been_filled = False
        
        print(f"  XBM初始化:")
        print(f"     - 队列大小: {memory_size}")
        print(f"     - 特征维度: {feat_dim}")
        print(f"     - 内存占用: ~{memory_size * feat_dim * 4 * 2 / (1024**2):.1f} MB (双队列)")
        print(f"     - Labels占用: ~{memory_size * 8 / (1024**2):.2f} MB")
        print(f"     - 设备: {device}")
    
    @property
    def is_full(self):
        """判断队列是否已满"""
        return self.has_been_filled
    
    def get(self):
        """
        获取队列中的所有有效特征和标签
        
        重要：返回的是引用/视图，不是拷贝！（零拷贝，高效）
        调用者必须确保在 enqueue_dequeue() 之前完成所有对返回数据的使用。
        
        正确使用顺序：
            1. xbm_feats = xbm.get()        # 获取引用
            2. loss = compute_loss(xbm_feats)  # 先计算损失
            3. xbm.enqueue_dequeue(...)     # 最后再入队
        
        Returns:
            feats1: 特征1队列 [N, feat_dim]，N <= memory_size （引用）
            feats2: 特征2队列 [N, feat_dim]，N <= memory_size （引用）
            labels: 标签队列 [N]，N <= memory_size （引用）
        """
        if self.is_full:
            # 队列已满，返回全部
            return self.feats1, self.feats2, self.labels
        else:
            # 队列未满，只返回已填充部分
            return self.feats1[:self.ptr], self.feats2[:self.ptr], self.labels[:self.ptr]
    
    def enqueue_dequeue(self, feats1, feats2, labels):
        """
        入队出队操作：将新特征和标签加入队列，移除最旧的（FIFO）
        
        核心操作：
        1. 将当前批次的特征和标签 detach 后存入队列
        2. 如果队列满了，最旧的特征会被新特征覆盖
        3. 使用循环队列结构
        
        Args:
            feats1: 当前批次特征1 [B, feat_dim]
            feats2: 当前批次特征2 [B, feat_dim]
            labels: 当前批次标签 [B]
        
        Note:
            - 特征必须 detach，防止梯度传播到历史特征
            - 使用原地操作，节省内存
        """
        batch_size = feats1.size(0)
        
        # 确保特征已detach（不参与梯度计算）
        feats1 = feats1.detach()
        feats2 = feats2.detach()
        
        # 确保labels是tensor且在正确的设备上
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        else:
            labels = labels.detach().to(self.device)
        
        # FIFO队列更新逻辑
        if self.ptr + batch_size > self.memory_size:
            # 当前批次会超出队列容量
            # 策略：将剩余空间填满，然后从队列头部继续填充
            remaining = self.memory_size - self.ptr
            
            # 填充剩余空间 [ptr:memory_size]
            self.feats1[self.ptr:] = feats1[:remaining]
            self.feats2[self.ptr:] = feats2[:remaining]
            self.labels[self.ptr:] = labels[:remaining]
            
            # 从队列头部填充溢出部分
            overflow = batch_size - remaining
            self.feats1[:overflow] = feats1[remaining:]  # 填充位置 0 到 overflow-1
            self.feats2[:overflow] = feats2[remaining:]
            self.labels[:overflow] = labels[remaining:]
            
            # 指针指向下一个空位
            self.ptr = overflow
            # 发生环绕，标记队列已满
            self.has_been_filled = True
        else:
            # 当前批次可以完全放入队列
            self.feats1[self.ptr: self.ptr + batch_size] = feats1
            self.feats2[self.ptr: self.ptr + batch_size] = feats2
            self.labels[self.ptr: self.ptr + batch_size] = labels
            self.ptr += batch_size
            # 刚好填满时也标记已满，避免is_full语义歧义
            if self.ptr >= self.memory_size:
                self.has_been_filled = True
    
    def get_size(self):
        """
        获取当前队列中有效特征的数量
        
        Returns:
            size: 队列中特征数量
        """
        if self.is_full:
            return self.memory_size
        else:
            return self.ptr
    
    def clear(self):
        """清空队列"""
        self.feats1.zero_()
        self.feats2.zero_()
        self.labels.zero_()
        self.ptr = 0
        self.has_been_filled = False
    
    def __repr__(self):
        return (f"XBM(memory_size={self.memory_size}, feat_dim={self.feat_dim}, "
                f"current_size={self.get_size()}, is_full={self.is_full})")
