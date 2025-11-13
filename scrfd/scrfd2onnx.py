"""
SCRFD ONNX 변환 스크립트
PyTorch 모델을 ONNX 형식으로 변환합니다.

필요한 패키지:
    pip install onnx onnxsim

사용 예시:
    # 기본 변환 (동적 입력 크기)
    python scrfd2onnx.py scrfd.pth
    
    # 고정 입력 크기
    python scrfd2onnx.py scrfd.pth --input-size 640 640 --no-dynamic
    
    # 출력 파일 지정
    python scrfd2onnx.py scrfd.pth --output scrfd_model.onnx
"""

import argparse
import torch
import torch.nn as nn
from scrfd_standalone import load_scrfd, SCRFD


class SCRFDWrapper(nn.Module):
    """ONNX export를 위한 SCRFD 래퍼.
    모델 출력을 리스트로 펼쳐서 반환합니다.
    """
    def __init__(self, model: SCRFD):
        super(SCRFDWrapper, self).__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor):
        """Forward pass.
        
        Args:
            x: Input tensor, shape (B, 3, H, W)
            
        Returns:
            outputs: Flattened list of outputs
                - cls_scores: [score_8, score_16, score_32]
                - bbox_preds: [bbox_8, bbox_16, bbox_32]
                - kps_preds: [kps_8, kps_16, kps_32] (if use_kps)
        """
        cls_scores, bbox_preds, kps_preds = self.model(x)
        
        # Flatten outputs for ONNX export
        outputs = []
        for cs in cls_scores:
            outputs.append(cs)
        for bp in bbox_preds:
            outputs.append(bp)
        if self.model.use_kps:
            for kp in kps_preds:
                outputs.append(kp)
        
        return tuple(outputs)


def export_onnx(checkpoint_path: str,
                output_path: str,
                input_shape: tuple = (640, 640),
                opset_version: int = 11,
                dynamic: bool = True,
                simplify: bool = True,
                device: str = 'cpu'):
    """SCRFD 모델을 ONNX로 변환.
    
    Args:
        checkpoint_path: PyTorch 체크포인트 경로
        output_path: 출력 ONNX 파일 경로
        input_shape: 입력 이미지 크기 (H, W), dynamic=True일 때는 더미 크기
        opset_version: ONNX opset 버전
        dynamic: 동적 입력 크기 지원 여부
        simplify: ONNX 모델 단순화 여부
        device: 변환 시 사용할 디바이스 ('cpu' 권장)
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # 모델 로드 (ONNX 변환은 CPU에서 하는 것이 안전)
    detector = load_scrfd(checkpoint_path, device=device)
    model = detector.model
    
    # 래퍼로 감싸기
    wrapped_model = SCRFDWrapper(model)
    wrapped_model.eval()
    
    # 더미 입력 생성
    if dynamic:
        # 동적 입력: batch, height, width는 동적
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
        print(f"Using dynamic input shape with dummy size: {input_shape}")
    else:
        # 고정 입력
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
        print(f"Using fixed input shape: {input_shape}")
    
    # 출력 이름 정의
    num_strides = len(model.strides)
    output_names = []
    
    # Classification scores
    for stride in model.strides:
        output_names.append(f'score_{stride}')
    
    # Bbox predictions
    for stride in model.strides:
        output_names.append(f'bbox_{stride}')
    
    # Keypoints (if enabled)
    if model.use_kps:
        for stride in model.strides:
            output_names.append(f'kps_{stride}')
    
    print(f"Output names: {output_names}")
    
    # 동적 축 정의
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'input.1': {
                0: 'batch',
                2: 'height',
                3: 'width'
            }
        }
        # 모든 출력에 대해 batch 차원 동적
        for out_name in output_names:
            dynamic_axes[out_name] = {
                0: 'batch',
                2: 'height',
                3: 'width'
            }
    
    # 임시 파일 경로
    if simplify or dynamic:
        temp_output = output_path.replace('.onnx', '_temp.onnx')
    else:
        temp_output = output_path
    
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    # ONNX export
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                dummy_input,
                temp_output,
                input_names=['input.1'],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                keep_initializers_as_inputs=False,
                verbose=False,
                do_constant_folding=True
            )
    except ModuleNotFoundError as e:
        if 'onnxscript' in str(e):
            print("\nError: onnxscript module is required for ONNX export.")
            print("Please install it with: pip install onnxscript")
            print("Or install all ONNX dependencies: pip install onnx onnxsim onnxscript")
            raise
        else:
            raise
    
    print(f"ONNX model exported to: {temp_output}")
    
    # Simplify (optional)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify
            
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(temp_output)
            
            if dynamic:
                # 동적 입력에 대한 simplify
                input_shapes = {
                    onnx_model.graph.input[0].name: [1, 3, input_shape[0], input_shape[1]]
                }
                simplified_model, check = simplify(
                    onnx_model,
                    input_shapes=input_shapes,
                    dynamic_input_shape=True
                )
            else:
                simplified_model, check = simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, output_path)
                print(f"Simplified ONNX model saved to: {output_path}")
                
                # 임시 파일 삭제
                if temp_output != output_path:
                    import os
                    os.remove(temp_output)
            else:
                print("Warning: Simplified model validation failed, using original model")
                if temp_output != output_path:
                    import shutil
                    shutil.move(temp_output, output_path)
        except ImportError:
            print("Warning: onnxsim not installed, skipping simplification")
            print("Install with: pip install onnxsim")
            if temp_output != output_path:
                import shutil
                shutil.move(temp_output, output_path)
        except Exception as e:
            print(f"Warning: Simplification failed: {e}")
            if temp_output != output_path:
                import shutil
                shutil.move(temp_output, output_path)
    else:
        if temp_output != output_path:
            import shutil
            shutil.move(temp_output, output_path)
    
    print(f"\nSuccessfully exported ONNX model: {output_path}")
    
    # 모델 정보 출력
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        print(f"\nModel Info:")
        print(f"  Input: {onnx_model.graph.input[0].name}")
        print(f"  Input shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
        print(f"  Outputs: {len(onnx_model.graph.output)}")
        for i, output in enumerate(onnx_model.graph.output):
            shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in output.type.tensor_type.shape.dim]
            print(f"    {output.name}: {shape}")
    except Exception as e:
        print(f"Could not load ONNX model info: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert SCRFD PyTorch model to ONNX')
    parser.add_argument('checkpoint', type=str, help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', '-o', type=str, default='', help='Output ONNX file path')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                       metavar=('H', 'W'), help='Input image size (default: 640 640)')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version (default: 11)')
    parser.add_argument('--no-dynamic', action='store_true', 
                       help='Disable dynamic input shape (use fixed size)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Disable ONNX model simplification')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for export (default: cpu)')
    
    args = parser.parse_args()
    
    # 출력 파일 경로 설정
    if not args.output:
        import os
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        if args.no_dynamic:
            output_name = f"{checkpoint_name}_shape{args.input_size[0]}x{args.input_size[1]}.onnx"
        else:
            output_name = f"{checkpoint_name}.onnx"
        args.output = output_name
    
    # 변환 실행
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_shape=tuple(args.input_size),
        opset_version=args.opset,
        dynamic=not args.no_dynamic,
        simplify=not args.no_simplify,
        device=args.device
    )


if __name__ == '__main__':
    main()

