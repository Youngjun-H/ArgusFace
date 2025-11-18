#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOLIDER 유틸리티 함수들
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = 'solider', level: int = logging.INFO) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


def get_model_name_from_path(model_path: str) -> str:
    """모델 파일 경로에서 모델 이름 추출"""
    model_path = Path(model_path)
    filename = model_path.stem.lower()
    
    # 모델 이름 매핑
    if 'swin_base' in filename:
        return 'swin_base'
    elif 'swin_small' in filename:
        return 'swin_small'
    elif 'swin_tiny' in filename:
        return 'swin_tiny'
    else:
        return 'swin_base'  # 기본값


def get_transformer_type(model_name: str) -> str:
    """모델 이름을 transformer 타입으로 변환"""
    mapping = {
        'swin_base': 'swin_base_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_tiny': 'swin_tiny_patch4_window7_224'
    }
    return mapping.get(model_name, 'swin_base_patch4_window7_224')


