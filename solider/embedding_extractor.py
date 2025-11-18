#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOLIDER 임베딩 추출기 - SOLIDER_REID 모델 전용
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .config import cfg, find_config_file, get_available_configs
from .model_factory import make_solider_model
from .utils import get_model_name_from_path, setup_logger


class SOLIDEREmbeddingExtractor:
    """
    SOLIDER_REID 모델을 사용하여 사람 이미지에서 ReID 임베딩을 추출하는 클래스
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        semantic_weight: float = 0.2,
        image_size: Tuple[int, int] = (384, 128),
        normalize_features: bool = True,
        config_path: Optional[str] = None,
    ):
        """
        SOLIDER 임베딩 추출기 초기화

        Args:
            model_path: 훈련된 모델 파일 경로
            device: 사용할 디바이스 ('cpu' 또는 'cuda')
            semantic_weight: 시맨틱 가중치
            image_size: 입력 이미지 크기 (height, width)
            normalize_features: 특징 벡터 정규화 여부
            config_path: 설정 파일 경로 (선택사항)
        """
        self.model_path = model_path
        self.device = device
        self.semantic_weight = semantic_weight
        self.image_size = image_size
        self.normalize_features = normalize_features

        # 로거 설정
        self.logger = setup_logger("solider_extractor")

        # 설정 로드
        self._load_config(config_path)

        # 모델 로드
        self._load_model()

        # 전처리 설정
        self._setup_transforms()

        self.logger.info("SOLIDER Embedding Extractor 초기화 완료")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Image size: {self.image_size}")
        self.logger.info(f"Semantic weight: {self.semantic_weight}")
        self.logger.info(f"Normalize features: {self.normalize_features}")

    def _load_config(self, config_path: Optional[str]):
        """설정 파일 로드"""
        if config_path and os.path.exists(config_path):
            cfg.merge_from_file(config_path)
            self.logger.info(f"설정 파일 로드: {config_path}")
        else:
            # 모델 이름으로 설정 파일 자동 찾기
            model_name = get_model_name_from_path(self.model_path)
            auto_config_path = find_config_file(model_name)

            if auto_config_path:
                cfg.merge_from_file(auto_config_path)
                self.logger.info(f"자동 설정 파일 로드: {auto_config_path}")
            else:
                self.logger.info("기본 설정 사용")
                available_configs = get_available_configs()
                if available_configs:
                    self.logger.info(f"사용 가능한 설정: {', '.join(available_configs)}")

        # 필수 설정 적용
        cfg.MODEL.semantic_weight = self.semantic_weight
        cfg.INPUT.size_train = list(self.image_size)
        cfg.INPUT.size_test = list(self.image_size)

    def _load_model(self):
        """모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # 모델 이름과 transformer 타입 결정
        model_name = get_model_name_from_path(self.model_path)
        from .utils import get_transformer_type

        transformer_type = get_transformer_type(model_name)

        # 체크포인트에서 클래스 수 자동 감지
        param_dict = torch.load(self.model_path, map_location=self.device)
        num_classes = self._detect_num_classes(param_dict)
        self.logger.info(f"감지된 클래스 수: {num_classes}")

        # 모델 생성 (감지된 클래스 수 사용)
        self.model = make_solider_model(
            transformer_type=transformer_type,
            num_classes=num_classes,  # 감지된 클래스 수 사용
            camera_num=6,  # 기본값
            view_num=1,  # 기본값
            semantic_weight=self.semantic_weight,
            img_size=self.image_size,
            drop_path=cfg.MODEL.drop_path,
            drop_out=cfg.MODEL.drop_out,
            att_drop_rate=cfg.MODEL.att_drop_rate,
        )

        # 가중치 로드
        try:
            self.model.load_state_dict(param_dict)
            self.logger.info(f"모델 로드 완료: {self.model_path}")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise

        # 디바이스로 이동
        self.model.to(self.device)
        self.model.eval()

    def _detect_num_classes(self, param_dict):
        """체크포인트에서 클래스 수 자동 감지"""
        # classifier.weight에서 클래스 수 감지
        if "classifier.weight" in param_dict:
            num_classes = param_dict["classifier.weight"].shape[0]
            self.logger.info(f"classifier.weight에서 클래스 수 감지: {num_classes}")
            return num_classes

        # 다른 가능한 키들 확인
        possible_keys = [
            "classifier.0.weight",  # 다른 구조일 수 있음
            "fc.weight",  # fully connected layer
            "head.weight",  # head layer
        ]

        for key in possible_keys:
            if key in param_dict:
                num_classes = param_dict[key].shape[0]
                self.logger.info(f"{key}에서 클래스 수 감지: {num_classes}")
                return num_classes

        # 기본값 반환
        self.logger.warning("클래스 수를 감지할 수 없어 기본값 1000 사용")
        return 1000

    def _setup_transforms(self):
        """이미지 전처리 변환 설정"""
        from torchvision import transforms

        # 정규화 파라미터
        mean = cfg.INPUT.pixel_mean
        std = cfg.INPUT.pixel_std

        self.transform = transforms.Compose(
            [transforms.Resize(self.image_size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

        self.logger.info("이미지 전처리 변환 설정 완료")

    def extract_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        단일 이미지에서 임베딩 추출

        Args:
            image: 이미지 (파일 경로, PIL Image, 또는 numpy 배열)

        Returns:
            임베딩 벡터 (numpy 배열)
        """
        # 이미지 로드 및 전처리
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        # 전처리
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 임베딩 추출
        with torch.no_grad():
            embedding = self.model(image_tensor)

            if self.normalize_features:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            embedding = embedding.cpu().numpy().flatten()

        return embedding

    def extract_embeddings_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """
        배치 이미지에서 임베딩 추출

        Args:
            images: 이미지 리스트

        Returns:
            임베딩 행렬 (numpy 배열)
        """
        embeddings = []

        for image in images:
            embedding = self.extract_embedding(image)
            embeddings.append(embedding)

        return np.array(embeddings)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 임베딩 간의 코사인 유사도 계산

        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩

        Returns:
            코사인 유사도 (0~1)
        """
        # L2 정규화
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 코사인 유사도 계산
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        임베딩 행렬의 유사도 행렬 계산

        Args:
            embeddings: 임베딩 행렬 (N x D)

        Returns:
            유사도 행렬 (N x N)
        """
        # 정규화
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        # 유사도 행렬 계산
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        return similarity_matrix
        return similarity_matrix
