# SCRFD ONNX 변환 가이드

## 필요한 패키지 설치

```bash
pip install onnx onnxsim onnxscript
```

또는

```bash
pip install onnx onnxsim
# PyTorch 2.0+ 버전의 경우 onnxscript도 필요할 수 있습니다
pip install onnxscript
```

## 사용 방법

### 기본 사용 (동적 입력 크기)

```bash
python scrfd2onnx.py scrfd.pth
```

이렇게 하면 `scrfd.onnx` 파일이 생성됩니다.

### 고정 입력 크기

```bash
python scrfd2onnx.py scrfd.pth --input-size 640 640 --no-dynamic
```

### 출력 파일 지정

```bash
python scrfd2onnx.py scrfd.pth --output scrfd_model.onnx
```

### 모든 옵션

```bash
python scrfd2onnx.py scrfd.pth \
    --output scrfd.onnx \
    --input-size 640 640 \
    --opset 11 \
    --no-dynamic \
    --no-simplify \
    --device cpu
```

## 옵션 설명

- `checkpoint`: PyTorch 체크포인트 파일 경로 (필수)
- `--output, -o`: 출력 ONNX 파일 경로 (기본값: 체크포인트 이름.onnx)
- `--input-size H W`: 입력 이미지 크기 (기본값: 640 640)
- `--opset`: ONNX opset 버전 (기본값: 11)
- `--no-dynamic`: 동적 입력 크기 비활성화 (고정 크기 사용)
- `--no-simplify`: ONNX 모델 단순화 비활성화
- `--device`: 변환 시 사용할 디바이스 (cpu 또는 cuda, 기본값: cpu)

## 출력 형식

ONNX 모델은 다음 출력을 생성합니다:

- `score_8`, `score_16`, `score_32`: 각 stride에 대한 classification scores
- `bbox_8`, `bbox_16`, `bbox_32`: 각 stride에 대한 bbox predictions
- `kps_8`, `kps_16`, `kps_32`: keypoints가 활성화된 경우 (선택적)

## ONNX Runtime으로 추론하기

```python
import onnxruntime as ort
import numpy as np
import cv2

# 모델 로드
session = ort.InferenceSession("scrfd.onnx")

# 이미지 전처리
img = cv2.imread("test.jpg")
img = cv2.resize(img, (640, 640))
img = (img - 127.5) / 128.0
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
img = np.expand_dims(img, axis=0).astype(np.float32)

# 추론
outputs = session.run(None, {"input.1": img})

# 출력 처리
# outputs[0]: score_8
# outputs[1]: score_16
# outputs[2]: score_32
# outputs[3]: bbox_8
# outputs[4]: bbox_16
# outputs[5]: bbox_32
```

