#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOLIDER 설정 시스템 - SOLIDER_REID 모델 설정
"""

import os
import yaml
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class ModelConfig:
    """모델 설정"""
    transformer_type: str = 'swin_base_patch4_window7_224'
    semantic_weight: float = 0.2
    drop_path: float = 0.1
    drop_out: float = 0.0
    att_drop_rate: float = 0.0


@dataclass
class InputConfig:
    """입력 설정"""
    size_train: List[int] = None
    size_test: List[int] = None
    pixel_mean: List[float] = None
    pixel_std: List[float] = None
    
    def __post_init__(self):
        if self.size_train is None:
            self.size_train = [384, 128]
        if self.size_test is None:
            self.size_test = [384, 128]
        if self.pixel_mean is None:
            self.pixel_mean = [0.485, 0.456, 0.406]
        if self.pixel_std is None:
            self.pixel_std = [0.229, 0.224, 0.225]


@dataclass
class TestConfig:
    """테스트 설정"""
    normalize_features: bool = True
    device: str = 'cpu'


@dataclass
class Config:
    """전체 설정"""
    MODEL: ModelConfig = None
    INPUT: InputConfig = None
    TEST: TestConfig = None
    
    def __post_init__(self):
        if self.MODEL is None:
            self.MODEL = ModelConfig()
        if self.INPUT is None:
            self.INPUT = InputConfig()
        if self.TEST is None:
            self.TEST = TestConfig()
    
    def merge_from_file(self, config_path: str):
        """YAML 파일에서 설정 로드"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 모델 설정 업데이트
        if 'MODEL' in config_dict:
            model_config = config_dict['MODEL']
            for key, value in model_config.items():
                if hasattr(self.MODEL, key.lower()):
                    setattr(self.MODEL, key.lower(), value)
        
        # 입력 설정 업데이트
        if 'INPUT' in config_dict:
            input_config = config_dict['INPUT']
            for key, value in input_config.items():
                if hasattr(self.INPUT, key.lower()):
                    setattr(self.INPUT, key.lower(), value)
        
        # 테스트 설정 업데이트
        if 'TEST' in config_dict:
            test_config = config_dict['TEST']
            for key, value in test_config.items():
                if hasattr(self.TEST, key.lower()):
                    setattr(self.TEST, key.lower(), value)


def find_config_file(model_name: str, dataset: str = 'msmt17') -> Optional[str]:
    """모델 이름에 따라 설정 파일을 자동으로 찾기"""
    # 설정 파일 디렉토리
    config_dir = os.path.join(os.path.dirname(__file__), 'configs', dataset)
    
    # 가능한 설정 파일 이름들
    possible_names = [
        f"{model_name}.yml",
        f"{model_name}.yaml",
        f"{model_name}_config.yml",
        f"{model_name}_config.yaml"
    ]
    
    for name in possible_names:
        config_path = os.path.join(config_dir, name)
        if os.path.exists(config_path):
            return config_path
    
    return None


def get_available_configs(dataset: str = 'msmt17') -> List[str]:
    """사용 가능한 설정 파일 목록 반환"""
    config_dir = os.path.join(os.path.dirname(__file__), 'configs', dataset)
    
    if not os.path.exists(config_dir):
        return []
    
    configs = []
    for file in os.listdir(config_dir):
        if file.endswith(('.yml', '.yaml')):
            configs.append(file.replace('.yml', '').replace('.yaml', ''))
    
    return sorted(configs)


def create_default_config() -> Config:
    """기본 설정 생성"""
    return Config()


# 전역 설정 인스턴스
cfg = Config()
