import os
from typing import Dict, List, Tuple

import numpy as np

# 기존 cross_camera_matching.py의 유틸리티 함수들 import
from cross_camera_matching import (
    MatchCandidate,
    TrackMeta,
    get_track_representative_image,
    load_track_embeddings,
    load_track_meta,
    lr_map_x,
    save_matched_pair_image,
)


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """
    embedding을 1D 배열로 정규화
    OSNet과 SOLIDER 모두 지원 (shape이 (D,), (1, D), (D, 1) 등 모두 처리)

    Args:
        emb: embedding 배열 (다양한 shape 가능)

    Returns:
        1D 배열 (D,)
    """
    emb = np.asarray(emb)
    emb = emb.flatten()  # 항상 1D로 변환
    return emb


def cross_camera_match_optimized(
    meta_dict: Dict[Tuple[str, str], TrackMeta],
    emb_dict: Dict[Tuple[str, str], np.ndarray],
    time_window_sec: float = 1.0,
    spatial_threshold: float = 0.2,
    sim_threshold: float = 0.5,
) -> List[MatchCandidate]:
    """
    최적화된 cross-camera matching 함수

    최적화 전략:
    1. 시간 기반 인덱싱: 시간 윈도우로 후보를 먼저 필터링
    2. 배치 행렬 연산: NumPy로 벡터화된 비교

    시간 복잡도: O(n×m) → O(n×k) (k << m, k는 시간 윈도우 내 track 수)

    Args:
        meta_dict: track 메타데이터 딕셔너리
        emb_dict: track embedding 딕셔너리
        time_window_sec: 시간 윈도우 (초)
        spatial_threshold: 공간 거리 임계값
        sim_threshold: similarity 임계값

    Returns:
        매칭 후보 리스트 (similarity 내림차순 정렬)
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

    # cam1 tracks를 시간 순으로 정렬 (시간 기반 인덱싱을 위해)
    cam1_tracks_sorted = sorted(cam1_tracks, key=lambda x: x[1].center_time)
    cam1_times = np.array([meta.center_time for _, meta in cam1_tracks_sorted])

    # cam1 embeddings를 미리 행렬로 변환 (배치 연산을 위해)
    # 모든 embedding을 1D로 정규화하여 일관성 유지
    cam1_embeddings_list = []
    for track, _ in cam1_tracks_sorted:
        emb = normalize_embedding(emb_dict[("cam1", track)])
        cam1_embeddings_list.append(emb)

    cam1_embeddings = np.array(cam1_embeddings_list)  # (M, D)

    candidates: List[MatchCandidate] = []

    # cam0 각 track에 대해
    for track0, meta0 in cam0_tracks:
        # emb0을 1D로 정규화
        emb0 = normalize_embedding(emb_dict[("cam0", track0)])  # (D,)
        t0 = meta0.center_time

        # 1) 시간 윈도우로 필터링 (이진 탐색 사용)
        # t0 - time_window_sec <= t1 <= t0 + time_window_sec
        time_lower = t0 - time_window_sec
        time_upper = t0 + time_window_sec

        # 이진 탐색으로 시간 범위 내의 인덱스 찾기
        left_idx = np.searchsorted(cam1_times, time_lower, side="left")
        right_idx = np.searchsorted(cam1_times, time_upper, side="right")

        if left_idx >= right_idx:
            continue  # 시간 윈도우 내에 cam1 track이 없음

        # 시간 윈도우 내의 cam1 tracks만 선택
        time_window_indices = np.arange(left_idx, right_idx)
        time_window_embeddings = cam1_embeddings[time_window_indices]  # (K, D), K = right_idx - left_idx
        time_window_tracks = [cam1_tracks_sorted[i] for i in time_window_indices]

        if len(time_window_embeddings) == 0:
            continue

        # 2) 배치 행렬 연산으로 embedding similarity 계산
        # emb0: (D,), time_window_embeddings: (K, D)
        # cosine similarity = dot(emb0, emb1) / (norm(emb0) * norm(emb1))
        # L2 normalized embedding이면 dot product = cosine similarity

        # emb0을 (1, D)로 확장
        emb0_expanded = np.expand_dims(emb0, 0)  # (1, D)

        # 배치 dot product: (1, D) @ (K, D).T = (1, K)
        similarities = np.dot(emb0_expanded, time_window_embeddings.T)[0]  # (K,)

        # 3) similarity 임계값 필터링
        valid_mask = similarities >= sim_threshold
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        # 4) 공간 조건 및 최종 후보 생성
        mapped_cx0 = lr_map_x(meta0.cx_norm)
        mapped_cy0 = meta0.cy_norm

        for idx in valid_indices:
            track1, meta1 = time_window_tracks[idx]
            sim = float(similarities[idx])
            t1 = meta1.center_time

            # 시간 차이
            dt = abs(t0 - t1)

            # 공간 거리 계산
            dx = abs(mapped_cx0 - meta1.cx_norm)
            dy = abs(mapped_cy0 - meta1.cy_norm)

            # 공간 필터링 (선택사항, 주석 해제하면 활성화)
            # spatial_dist = max(dx, dy)
            # if spatial_dist > spatial_threshold:
            #     continue

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
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Optimized cross-camera matching")
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="track_embeddings_osnet",
        help="Directory containing track embeddings (default: track_embeddings_osnet)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for match results (default: match_results_{embedding_dir})",
    )
    parser.add_argument(
        "--time_window",
        type=float,
        default=10.0,
        help="Time window in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--spatial_threshold",
        type=float,
        default=0.2,
        help="Spatial distance threshold (default: 0.2)",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.5,
        help="Similarity threshold (default: 0.5)",
    )
    args = parser.parse_args()

    # 1) 메타데이터 / 임베딩 로드
    print("[INFO] Loading metadata and embeddings...")
    meta_dict = load_track_meta("track_meta.csv")
    emb_dict = load_track_embeddings(args.embedding_dir)
    print(f"[INFO] Loaded {len(meta_dict)} tracks, {len(emb_dict)} embeddings")

    # 첫 번째 embedding의 shape 확인 (디버깅용)
    if emb_dict:
        first_emb = list(emb_dict.values())[0]
        print(f"[INFO] Embedding shape: {first_emb.shape} (normalized to 1D)")

    # 2) 최적화된 매칭 실행
    print("\n[INFO] Running optimized cross-camera matching...")
    start_time = time.time()

    matches = cross_camera_match_optimized(
        meta_dict,
        emb_dict,
        time_window_sec=args.time_window,
        spatial_threshold=args.spatial_threshold,
        sim_threshold=args.sim_threshold,
    )

    elapsed_time = time.time() - start_time
    print(f"[INFO] Matching completed in {elapsed_time:.2f} seconds")
    print(f"[INFO] Found {len(matches)} matches")

    # 3) 결과 출력 및 매칭 이미지 저장
    print("\n==== Cross-camera match candidates ====")
    if args.output_dir is None:
        # embedding_dir에서 자동으로 output_dir 생성
        output_match_dir = f"match_results_{os.path.basename(args.embedding_dir)}"
    else:
        output_match_dir = args.output_dir
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
