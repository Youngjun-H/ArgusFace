import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from natsort import natsorted
from torchvision import transforms

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from osnet import OSNetFeatureExtractor  # noqa: E402
from osnet import compute_pairwise_similarity
from solider import SOLIDEREmbeddingExtractor


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_image_files(folder, exts=(".jpg", ".jpeg", ".png")):
    return natsorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


def extract_embeddings_for_segments(
    segments_base_dir="segments",
    output_tracklet_embedding_dir="track_embeddings",
    image_exts=(".jpg", ".jpeg", ".png"),
    device="cuda",
):
    """
    segments/ 구조:
      segments/cam0/track_1/seg_000_best.jpg
      segments/cam0/track_1/seg_001_best.jpg
      segments/cam1/track_3/seg_000_best.jpg
      ...

    하는 일:
      1) 각 seg_*_best.jpg 에 대해 OSNet ReID embedding 계산
         -> 같은 폴더에 seg_*_best.npy 로 저장
      2) 각 track_*/ 에 있는 모든 segment embedding 의 평균을
         track_embeddings/camX/track_Y.npy 로 저장
    """

    extractor = OSNetFeatureExtractor(
        model_name="osnet_ain_x1_0",
        model_path="/home/yjhwang/work/argusface/osnet/checkpoints/osnet_ain_x1_0_msmt17.pth",  # update path
        device="cuda",
    )

    solider_extractor = SOLIDEREmbeddingExtractor(
        model_path="/home/yjhwang/work/argusface/solider/checkpoints/swin_base_msmt17.pth",
        config_path="models/solider/configs/msmt17/swin_base.yml",
        device="cuda",
        semantic_weight=0.2,
        image_size=(384, 128),
        normalize_features=True,
    )

    for cam_name in natsorted(os.listdir(segments_base_dir)):
        cam_path = os.path.join(segments_base_dir, cam_name)
        if not os.path.isdir(cam_path):
            continue

        print(f"[CAM] {cam_name}")

        for track_name in natsorted(os.listdir(cam_path)):
            track_path = os.path.join(cam_path, track_name)
            if not os.path.isdir(track_path):
                continue

            print(f"  [TRACK] {cam_name}/{track_name}")

            seg_files = list_image_files(track_path, exts=image_exts)
            if not seg_files:
                print("    -> No segment images, skip.")
                continue

            # Tracklet centroid를 만들기 위한 embedding 리스트
            track_embeddings = []

            for seg_img_name in seg_files:
                seg_img_path = os.path.join(track_path, seg_img_name)

                # npy가 이미 있으면 스킵하고 로드해도 됨
                seg_base, _ = os.path.splitext(seg_img_name)
                seg_npy_path = os.path.join(track_path, seg_base + ".npy")

                # 이미지 로드
                img = cv2.imread(seg_img_path)
                if img is None:
                    print(f"    [WARN] Failed to read image: {seg_img_path}")
                    continue

                # ReID embedding 추출
                emb = extractor(img)  # (D,)
                # emb = solider_extractor.extract_embedding(img)  # numpy array 반환

                # numpy 배열을 텐서로 변환 (torch.stack을 위해)
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb).to(device)

                # segment-level embedding 저장
                np.save(seg_npy_path, emb.cpu().numpy())
                track_embeddings.append(emb)

                print(f"    [SEG] {seg_img_name} -> emb shape={emb.shape}, saved={seg_npy_path}")

            if len(track_embeddings) == 0:
                print("    -> No valid embedding for this track, skip centroid.")
                continue

            # Tracklet centroid embedding 계산 (단순 평균)
            # CUDA 텐서를 numpy로 변환하기 전에 torch.stack 사용
            track_embeddings_tensor = torch.stack(track_embeddings, dim=0)  # (N_segments, D)
            centroid_tensor = track_embeddings_tensor.mean(dim=0)  # (D,)

            # centroid도 L2 normalize (옵션이지만 추천)
            centroid_tensor = centroid_tensor / (torch.norm(centroid_tensor) + 1e-12)

            # CPU로 이동 후 numpy로 변환
            centroid = centroid_tensor.cpu().numpy()

            # tracklet centroid 저장
            out_tracklet_dir = os.path.join(output_tracklet_embedding_dir, cam_name)
            ensure_dir(out_tracklet_dir)

            out_tracklet_path = os.path.join(out_tracklet_dir, track_name + ".npy")
            np.save(out_tracklet_path, centroid)

            print(f"  [TRACK CENTROID] {cam_name}/{track_name} -> {out_tracklet_path}, shape={centroid.shape}")


# ============================================================
# 4. 실행부
# ============================================================

if __name__ == "__main__":
    """
    segments/ 에서 시작해서:
      - segment-level embedding (각 seg_XXX_best.npy)
      - track-level centroid embedding (track_embeddings/camX/track_Y.npy)
    를 생성한다.
    """
    extract_embeddings_for_segments(
        segments_base_dir="segments",
        output_tracklet_embedding_dir="track_embeddings_osnet",
        device="cuda",  # cuda / cpu 중 선택
    )
