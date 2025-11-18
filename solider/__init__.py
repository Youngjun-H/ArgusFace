# SOLIDER_REID 모델 패키지
from .embedding_extractor import SOLIDEREmbeddingExtractor
from .model_factory import make_solider_model, SOLIDERModel
from .config import Config, create_default_config
from .utils import setup_logger, get_model_name_from_path, get_transformer_type
from .swin_transformer import *

__all__ = [
    'SOLIDEREmbeddingExtractor',
    'make_solider_model',
    'SOLIDERModel',
    'Config',
    'create_default_config',
    'setup_logger',
    'get_model_name_from_path',
    'get_transformer_type',
    'swin_base_patch4_window7_224',
    'swin_small_patch4_window7_224', 
    'swin_tiny_patch4_window7_224',
    'SwinTransformer',
    'PatchEmbed',
    'PatchMerging',
    'WindowMSA',
    'ShiftWindowMSA',
    'SwinBlock',
    'SwinBlockSequence'
]
