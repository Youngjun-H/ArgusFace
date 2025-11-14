import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


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
            spatial_dist = max(dx, dy)
            if spatial_dist > spatial_threshold:
                continue

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
        time_window_sec=1.0,
        spatial_threshold=0.2,
        sim_threshold=0.55,
    )

    # 3) 결과 출력
    print("==== Cross-camera match candidates ====")
    for m in matches:
        print(
            f"cam0/{m.cam0_track}  <-->  cam1/{m.cam1_track} | "
            f"sim={m.sim:.3f}, dt={m.dt:.2f}s, dx={m.dx:.3f}, dy={m.dy:.3f}"
        )
