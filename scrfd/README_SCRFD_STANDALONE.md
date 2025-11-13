# SCRFD Standalone Implementation

순수 PyTorch + OpenCV로 구현된 SCRFD 얼굴 검출 모델입니다. mmdetection과 mmcv 의존성 없이 사용할 수 있습니다.

## 의존성

- `torch` (PyTorch)
- `opencv-python` (cv2)
- `numpy`

## 설치

```bash
pip install torch opencv-python numpy
```

## 사용법

### 기본 사용

```python
import cv2
from scrfd_standalone import load_scrfd

# 모델 로드
detector = load_scrfd('scrfd.pth', device='cuda')

# 이미지 로드 및 검출
img = cv2.imread('test.jpg')
detections, keypoints = detector.detect(img, thresh=0.5)

# 결과 출력
for det in detections:
    x1, y1, x2, y2, score = det
    print(f"Face: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), score={score:.3f}")
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

cv2.imwrite('result.jpg', img)
```

### 커맨드라인 사용

```bash
python scrfd_standalone.py scrfd.pth test.jpg
```

### 커스텀 설정

다른 모델 크기나 설정을 사용하는 경우:

```python
config = {
    'backbone': {
        'depth': 56,  # 또는 다른 depth
        'base_channels': 64,
        'num_stages': 4,
        'out_indices': (0, 1, 2, 3),
        'strides': (1, 2, 2, 2),
        'avg_down': True,
        'deep_stem': True
    },
    'neck': {
        'in_channels': [64, 512, 1024, 2048],
        'out_channels': 128,
        'num_outs': 3,
        'start_level': 1,
        'add_extra_convs': 'on_output'
    },
    'head': {
        'num_classes': 1,
        'in_channels': 128,
        'stacked_convs': 2,
        'feat_channels': 256,
        'num_groups': 32,
        'cls_reg_share': True,
        'strides_share': True,
        'scale_mode': 2,
        'use_dfl': False,
        'reg_max': 8,
        'use_kps': False,  # 키포인트 사용 시 True
        'num_anchors': 2,
        'strides': [8, 16, 32]
    }
}

detector = load_scrfd('scrfd.pth', config=config, device='cuda')
```

## 모델 구조

- **Backbone**: ResNetV1e (BasicBlock 또는 Bottleneck)
- **Neck**: PAFPN (Path Aggregation FPN)
- **Head**: SCRFDHead (분류, 회귀, 키포인트)

## 주요 기능

1. **의존성 최소화**: mmdetection/mmcv 없이 동작
2. **간단한 API**: `detect()` 메서드로 쉽게 사용
3. **유연한 설정**: 다양한 모델 크기 지원
4. **키포인트 지원**: 옵션으로 5개 키포인트 예측 가능

## 체크포인트 로딩

체크포인트 파일의 키 이름이 다를 수 있습니다. 필요시 `load_scrfd` 함수에서 키 매핑을 수정하세요.

## 주의사항

- 체크포인트의 모델 구조와 설정이 일치해야 합니다
- 전처리 정규화: mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]
- 입력 이미지는 BGR 형식이어야 합니다

