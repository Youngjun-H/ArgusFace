import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackMeta:
    cam: str  # "cam0" or "cam1"
    track: str  # "track_1", ...
    start_time: float  # seconds
    end_time: float  # seconds
    cx_norm: float  # 0~1
    cy_norm: float  # 0~1

    @property
    def center_time(self) -> float:
        return 0.5 * (self.start_time + self.end_time)


def load_track_meta(csv_path: str) -> Dict[Tuple[str, str], TrackMeta]:
    """
    track_meta.csv 형식:

    cam,track,start_time,end_time,cx_norm,cy_norm
    cam0,track_1,0.5,4.2,0.8,0.6
    cam1,track_3,0.7,4.0,0.15,0.6
    ...

    반환:
      key: (cam, track)
      value: TrackMeta
    """
    meta_dict: Dict[Tuple[str, str], TrackMeta] = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam = row["cam"]
            track = row["track"]
            start_time = float(row["start_time"])
            end_time = float(row["end_time"])
            cx_norm = float(row["cx_norm"])
            cy_norm = float(row["cy_norm"])

            meta = TrackMeta(
                cam=cam,
                track=track,
                start_time=start_time,
                end_time=end_time,
                cx_norm=cx_norm,
                cy_norm=cy_norm,
            )
            meta_dict[(cam, track)] = meta

    return meta_dict


def load_track_embeddings(emb_base_dir: str) -> Dict[Tuple[str, str], np.ndarray]:
    """
    track_embeddings/ 구조:

      track_embeddings/
        cam0/
          track_1.npy
          track_2.npy
        cam1/
          track_3.npy
          track_5.npy

    반환:
      key: (cam, track)
      value: embedding (np.ndarray, shape=(D,))
    """
    emb_dict: Dict[Tuple[str, str], np.ndarray] = {}

    for cam_name in os.listdir(emb_base_dir):
        cam_dir = os.path.join(emb_base_dir, cam_name)
        if not os.path.isdir(cam_dir):
            continue

        for fname in os.listdir(cam_dir):
            if not fname.endswith(".npy"):
                continue

            track_name = os.path.splitext(fname)[0]  # "track_1"

            emb_path = os.path.join(cam_dir, fname)
            emb = np.load(emb_path)  # (D,)

            emb_dict[(cam_name, track_name)] = emb

    return emb_dict


def lr_map_x(cam0_cx_norm: float) -> float:
    """
    cam0 의 x 좌표를 cam1 관점으로 좌우 반전 매핑한다고 가정.
    0~1 정규화 좌표에서 단순히 1 - x.
    """
    return 1.0 - cam0_cx_norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # 1D 배열로 변환 (shape이 (1, D) 또는 (D,) 모두 처리)
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def get_track_representative_image(
    tracklets_base_dir: str, segments_base_dir: Optional[str], cam: str, track: str
) -> Optional[np.ndarray]:
    """
    track의 대표 이미지를 가져옴
    1. segments가 있으면 seg_000_best.jpg 우선 사용
    2. 없으면 tracklets에서 첫 번째 이미지 사용

    Returns:
        이미지 (BGR) 또는 None
    """
    # 1. segments에서 찾기
    if segments_base_dir:
        seg_path = os.path.join(segments_base_dir, cam, track, "seg_000_best.jpg")
        if os.path.exists(seg_path):
            img = cv2.imread(seg_path)
            if img is not None:
                return img

    # 2. tracklets에서 첫 번째 이미지 찾기
    tracklet_dir = os.path.join(tracklets_base_dir, cam, track)
    if not os.path.isdir(tracklet_dir):
        return None

    # 이미지 파일 찾기
    image_files = sorted([f for f in os.listdir(tracklet_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not image_files:
        return None

    first_img_path = os.path.join(tracklet_dir, image_files[0])
    img = cv2.imread(first_img_path)
    return img


def save_matched_pair_image(
    img0: np.ndarray,
    img1: np.ndarray,
    output_dir: str,
    cam0_track: str,
    cam1_track: str,
    sim: float,
    match_idx: int,
):
    """
    두 이미지를 나란히 붙여서 저장

    Args:
        img0: cam0 이미지
        img1: cam1 이미지
        output_dir: 출력 디렉토리
        cam0_track: cam0 track 이름
        cam1_track: cam1 track 이름
        sim: similarity 점수
        match_idx: 매칭 인덱스
    """
    os.makedirs(output_dir, exist_ok=True)

    # 두 이미지의 높이를 맞춤 (더 큰 높이에 맞춤)
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    max_h = max(h0, h1)

    # 높이 맞추기 (비율 유지하며 리사이즈)
    if h0 != max_h:
        scale = max_h / h0
        new_w0 = int(w0 * scale)
        img0 = cv2.resize(img0, (new_w0, max_h))
    if h1 != max_h:
        scale = max_h / h1
        new_w1 = int(w1 * scale)
        img1 = cv2.resize(img1, (new_w1, max_h))

    # 두 이미지를 가로로 붙이기
    combined = np.hstack([img0, img1])

    # 텍스트 추가 (선택사항)
    h, w = combined.shape[:2]
    text = f"{cam0_track} <-> {cam1_track} | sim={sim:.3f}"
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 파일명: match_001_cam0_track_1_cam1_track_3_sim0.711.jpg
    filename = f"match_{match_idx:03d}_{cam0_track}_{cam1_track}_sim{sim:.3f}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, combined)


@dataclass
class MatchCandidate:
    cam0_track: str
    cam1_track: str
    sim: float
    dt: float
    dx: float
    dy: float


def cross_camera_match(
    meta_dict: Dict[Tuple[str, str], TrackMeta],
    emb_dict: Dict[Tuple[str, str], np.ndarray],
    time_window_sec: float = 1.0,
    spatial_threshold: float = 0.2,
    sim_threshold: float = 0.5,
) -> List[MatchCandidate]:
    """
    cam0 / cam1 의 track embeddings + meta 를 기반으로
    시간 / 공간 / ReID sim 조건을 모두 만족하는 매칭 후보 리스트를 반환.

    - time_window_sec: 중심 시간 차이 절대값 <= 이 값일 때 후보
    - spatial_threshold: 좌우 보정 후 x,y 거리 <= 이 값일 때 후보 (정규화 좌표기준)
    - sim_threshold: cosine similarity >= 이 값일 때 최종 매칭 후보
    """

    # cam0 / cam1 track 분리
    cam0_tracks = []
    cam1_tracks = []

    for (cam, track), meta in meta_dict.items():
        if (cam, track) not in emb_dict:
            continue  # embedding 없는 경우 skip

        if cam == "cam0":
            cam0_tracks.append((track, meta))
        elif cam == "cam1":
            cam1_tracks.append((track, meta))

    candidates: List[MatchCandidate] = []

    # 모든 cam0-track / cam1-track 쌍을 비교 (나중에 효율화 가능)
    for track0, meta0 in cam0_tracks:
        emb0 = emb_dict[("cam0", track0)]
        t0 = meta0.center_time

        for track1, meta1 in cam1_tracks:
            emb1 = emb_dict[("cam1", track1)]
            t1 = meta1.center_time

            # 1) Temporal window 조건
            dt = abs(t0 - t1)
            if dt > time_window_sec:
                continue

            # 2) 좌우 보정 + Spatial 조건
            mapped_cx0 = lr_map_x(meta0.cx_norm)
            mapped_cy0 = meta0.cy_norm  # y는 그대로 사용 (필요하면 보정 추가 가능)

            dx = abs(mapped_cx0 - meta1.cx_norm)
            dy = abs(mapped_cy0 - meta1.cy_norm)

            # 거리 기준 (단순 L-infinity 혹은 L2 사용 가능)
            # spatial_dist = max(dx, dy)
            # if spatial_dist > spatial_threshold:
            #     continue

            # 3) Embedding similarity 조건
            sim = cosine_similarity(emb0, emb1)
            if sim < sim_threshold:
                continue

            # 모든 조건 통과 → 매칭 후보
            candidates.append(
                MatchCandidate(
                    cam0_track=track0,
                    cam1_track=track1,
                    sim=sim,
                    dt=dt,
                    dx=dx,
                    dy=dy,
                )
            )

    # similarity 내림차순으로 정렬
    candidates.sort(key=lambda x: x.sim, reverse=True)

    return candidates


if __name__ == "__main__":
    # 1) 메타데이터 / 임베딩 로드
    meta_dict = load_track_meta("track_meta.csv")
    emb_dict = load_track_embeddings("track_embeddings")

    # 2) 매칭 실행
    matches = cross_camera_match(
        meta_dict,
        emb_dict,
        time_window_sec=10.0,
        spatial_threshold=0.2,
        sim_threshold=0.5,
    )

    # 3) 결과 출력 및 매칭 이미지 저장
    print("==== Cross-camera match candidates ====")
    output_match_dir = "match_results"
    os.makedirs(output_match_dir, exist_ok=True)

    tracklets_base_dir = "tracklets"
    segments_base_dir = "segments" if os.path.exists("segments") else None

    for idx, m in enumerate(matches, 1):
        print(
            f"cam0/{m.cam0_track}  <-->  cam1/{m.cam1_track} | "
            f"sim={m.sim:.3f}, dt={m.dt:.2f}s, dx={m.dx:.3f}, dy={m.dy:.3f}"
        )

        # 매칭된 이미지 가져오기
        img0 = get_track_representative_image(tracklets_base_dir, segments_base_dir, "cam0", m.cam0_track)
        img1 = get_track_representative_image(tracklets_base_dir, segments_base_dir, "cam1", m.cam1_track)

        if img0 is not None and img1 is not None:
            save_matched_pair_image(img0, img1, output_match_dir, m.cam0_track, m.cam1_track, m.sim, idx)
            print(f"  -> Saved: match_{idx:03d}_{m.cam0_track}_{m.cam1_track}_sim{m.sim:.3f}.jpg")
        else:
            print(f"  -> Warning: Could not load images for {m.cam0_track} or {m.cam1_track}")

    print(f"\n[INFO] Total {len(matches)} matches found. Images saved to: {output_match_dir}/")
