#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOLIDER 모델 팩토리 - SOLIDER_REID 모델 생성
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .swin_transformer import (
    swin_base_patch4_window7_224,
    swin_small_patch4_window7_224, 
    swin_tiny_patch4_window7_224
)

logger = logging.getLogger(__name__)


def weights_init_xavier(m):
    """Xavier 초기화"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class SOLIDERModel(nn.Module):
    """SOLIDER_REID 모델"""
    
    def __init__(self, 
                 transformer_type: str,
                 num_classes: int,
                 camera_num: int,
                 view_num: int,
                 semantic_weight: float,
                 img_size: tuple = (384, 128),
                 drop_path: float = 0.1,
                 drop_out: float = 0.0,
                 att_drop_rate: float = 0.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.transformer_type = transformer_type
        self.semantic_weight = semantic_weight
        
        # Transformer 팩토리
        transformer_factory = {
            'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
            'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
            'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
        }
        
        if transformer_type not in transformer_factory:
            raise ValueError(f"Unsupported transformer type: {transformer_type}")
        
        # Transformer 백본 생성
        self.base = transformer_factory[transformer_type](
            img_size=img_size,
            drop_rate=drop_out,
            attn_drop_rate=att_drop_rate,
            drop_path_rate=drop_path,
            semantic_weight=semantic_weight
        )
        
        # 분류기 (SwinTransformer의 마지막 출력 차원 사용)
        embed_dim = self.base.num_features[-1]
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        
        # 드롭아웃
        self.dropout = nn.Dropout(drop_out)
        
        # 가중치 초기화
        self.apply(weights_init_xavier)
    
    def forward(self, x, return_feat=False):
        """순전파"""
        # 백본에서 특징 추출 (SwinTransformer는 tuple을 반환하므로 첫 번째 요소만 사용)
        base_output = self.base(x)
        if isinstance(base_output, tuple):
            global_feat = base_output[0]  # 첫 번째 요소만 사용
        else:
            global_feat = base_output
        
        # 배치 정규화
        feat = self.bottleneck(global_feat)
        
        if self.training:
            # 훈련 시: 분류기 출력
            cls_score = self.classifier(self.dropout(feat))
            return cls_score
        else:
            # 추론 시: 특징 벡터 반환
            if return_feat:
                return global_feat, feat
            else:
                return feat
    
    def to(self, device):
        """모델을 지정된 디바이스로 이동"""
        super().to(device)
        # 모든 하위 모듈도 같은 디바이스로 이동
        self.base.to(device)
        self.classifier.to(device)
        self.bottleneck.to(device)
        self.dropout.to(device)
        return self


def make_solider_model(transformer_type: str,
                      num_classes: int,
                      camera_num: int,
                      view_num: int,
                      semantic_weight: float,
                      img_size: tuple = (384, 128),
                      drop_path: float = 0.1,
                      drop_out: float = 0.0,
                      att_drop_rate: float = 0.0) -> SOLIDERModel:
    """SOLIDER 모델 생성"""
    return SOLIDERModel(
        transformer_type=transformer_type,
        num_classes=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=semantic_weight,
        img_size=img_size,
        drop_path=drop_path,
        drop_out=drop_out,
        att_drop_rate=att_drop_rate
    )
