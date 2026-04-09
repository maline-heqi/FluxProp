import torch
import torch.nn as nn

def freeze_all_layers_except(model: nn.Module, layer_idx: int):
    """
    针对 FluxProp 架构量身定制的逐层冻结工具函数。
    确保在深层时空推演中，梯度流不会意外污染已冻结的底层结构。
    """
    
    # ==========================================
    # 1. 物理断开全局梯度流 (一键全部锁死)
    # ==========================================
    for param in model.parameters():
        param.requires_grad = False

    # ==========================================
    # 2. 精准点亮目标层 
    # (解冻当前层的所有物理组件：U, W, W_Q, W_K, W_V, W_O, alpha, gamma)
    # ==========================================
    if 0 <= layer_idx < len(model.layers):
        for param in model.layers[layer_idx].parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"❌ 严重错误: layer_idx {layer_idx} 超出模型实际层数范围 (0 - {len(model.layers)-1})")

    # ==========================================
    # 3. 架构级特判：Embedding 与 LM_Head 的联合空间
    # ==========================================
    if layer_idx == 0:
        # 训练最底层时，必须允许更新词典的 Embedding 空间，否则底层没有特征来源。
        # 由于在定义时写了 `self.lm_head.weight = self.embedding.weight`，
        # 这里解冻 embedding 的同时，lm_head 的权重也自动获得了梯度。
        for param in model.embedding.parameters():
            param.requires_grad = True
        
        # 不要遗漏独立的 bias 偏置项
        if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
            model.lm_head.bias.requires_grad = True
            
        print(f"🔓 [梯度控制] 训练 Layer 0: 已激活 Layer 0 物理层 + Embedding/LM_Head 联合词表空间")
    else:
        # 训练高层时，底层 Embedding 空间已经被 Layer 0 固化。
        # 高层只需要学会如何将复杂的特征映射回这个已固化的词表空间中。
        print(f"🔒 [梯度控制] 训练 Layer {layer_idx}: 已物理锁死底层与词表空间，仅计算当前层时空梯度")

    # ==========================================
    # 4. 工程自检：打印实际激活的参数规模
    # ==========================================
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params
    
    print(f"⚙️  模型参数审计: 总计 {total_params:,} | 激活 {trainable_params:,} (约 {trainable_params/total_params*100:.1f}%) | 冻结 {frozen_params:,}")