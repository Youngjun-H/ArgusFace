"""
SCRFD 디버깅 스크립트
체크포인트 키 매핑 및 모델 출력 확인
"""

import torch
import cv2
import numpy as np
from scrfd_standalone import load_scrfd

def check_checkpoint_keys(checkpoint_path):
    """체크포인트의 키 구조 확인"""
    print("=" * 60)
    print("Checking checkpoint keys...")
    print("=" * 60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Checkpoint has 'state_dict' key")
    else:
        state_dict = checkpoint
        print(f"Checkpoint is directly a state_dict")
    
    print(f"\nTotal keys: {len(state_dict)}")
    print("\nFirst 30 keys:")
    for i, key in enumerate(list(state_dict.keys())[:30]):
        print(f"  {i+1:2d}. {key}")
    
    # 키 패턴 분석
    patterns = {}
    for key in state_dict.keys():
        prefix = key.split('.')[0]
        patterns[prefix] = patterns.get(prefix, 0) + 1
    
    print("\nKey patterns (prefix counts):")
    for prefix, count in sorted(patterns.items()):
        print(f"  {prefix}: {count} keys")
    
    return state_dict

def test_model_outputs(detector, img_path):
    """모델 출력 확인"""
    print("\n" + "=" * 60)
    print("Testing model outputs...")
    print("=" * 60)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Preprocess
    tensor, det_scale = detector.preprocess(img)
    tensor = tensor.to(detector.device)
    print(f"Preprocessed tensor shape: {tensor.shape}")
    print(f"Det scale: {det_scale:.4f}")
    
    # Forward
    with torch.no_grad():
        outputs = detector.model(tensor)
    
    cls_scores, bbox_preds, kps_preds = outputs
    
    print(f"\nNumber of output levels: {len(cls_scores)}")
    for i, (cs, bp) in enumerate(zip(cls_scores, bbox_preds)):
        print(f"Level {i} (stride {detector.strides[i]}):")
        print(f"  cls_score shape: {cs.shape}")
        print(f"  bbox_pred shape: {bp.shape}")
        print(f"  cls_score range: [{cs.min().item():.4f}, {cs.max().item():.4f}]")
        print(f"  cls_score after sigmoid max: {torch.sigmoid(cs[0]).max().item():.4f}")
    
    # Test with different thresholds
    print("\n" + "=" * 60)
    print("Testing with different thresholds...")
    print("=" * 60)
    
    for thresh in [0.01, 0.02, 0.05, 0.1, 0.3, 0.5]:
        detections, keypoints = detector.detect(img, thresh=thresh)
        print(f"Threshold {thresh:.2f}: {len(detections)} detections")
        if len(detections) > 0:
            print(f"  Top 3 scores: {detections[:min(3, len(detections)), 4]}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_scrfd_debug.py <checkpoint_path> [image_path]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else 'test/people.jpg'
    
    # Check checkpoint keys
    state_dict = check_checkpoint_keys(checkpoint_path)
    
    # Load model
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)
    detector = load_scrfd(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test outputs
    if image_path:
        test_model_outputs(detector, image_path)

