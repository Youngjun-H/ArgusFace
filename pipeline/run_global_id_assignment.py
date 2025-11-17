############################################################
# run_global_id_assignment.py
#
# 전체 파이프라인:
# 1) track_embeddings 로드
# 2) track_meta.csv 로드
# 3) cross-camera matching 실행
# 4) Global ID Manager로 global_id 할당
# 5) 매칭 안된 track들에도 global_id 부여
# 6) global_id_map.json 저장
############################################################

import os

from cross_camera_matching import cross_camera_match, load_track_embeddings, load_track_meta
from global_id_manager import GlobalIDManager

############################################################
# 0. 헬퍼 함수: 트랙 리스트 읽기
############################################################


def get_all_tracks(emb_base_dir):
    """
    track_embeddings/
        cam0/
            track_1.npy
            track_2.npy
        cam1/
            track_3.npy
            track_7.npy
    위와 같은 구조에서 cam0, cam1의 모든 track 리스트 추출
    """
    cam0_tracks = []
    cam1_tracks = []

    cam0_dir = os.path.join(emb_base_dir, "cam0")
    cam1_dir = os.path.join(emb_base_dir, "cam1")

    if os.path.isdir(cam0_dir):
        for f in os.listdir(cam0_dir):
            if f.endswith(".npy"):
                cam0_tracks.append(os.path.splitext(f)[0])  # "track_1"

    if os.path.isdir(cam1_dir):
        for f in os.listdir(cam1_dir):
            if f.endswith(".npy"):
                cam1_tracks.append(os.path.splitext(f)[0])  # "track_3"

    return cam0_tracks, cam1_tracks


############################################################
# 1. Main
############################################################


def main():
    # --------------------------------------------------------
    # Step 1) track_meta.csv 로드
    # --------------------------------------------------------
    meta_dict = load_track_meta("track_meta.csv")

    # --------------------------------------------------------
    # Step 2) track_embeddings 로드
    # --------------------------------------------------------
    emb_dict = load_track_embeddings("track_embeddings")

    # --------------------------------------------------------
    # Step 3) cross-camera matching (matching candidates 리스트 생성)
    # --------------------------------------------------------
    matches = cross_camera_match(
        meta_dict,
        emb_dict,
        time_window_sec=1.0,
        spatial_threshold=0.2,
        sim_threshold=0.55,
    )

    print(f"\n[INFO] Matching candidates found: {len(matches)}")
    for m in matches[:10]:
        print(f"  cam0/{m.cam0_track} <-> cam1/{m.cam1_track}, sim={m.sim:.3f}")

    # --------------------------------------------------------
    # Step 4) GlobalIDManager 생성
    # --------------------------------------------------------
    gid_manager = GlobalIDManager(save_path="global_id_map.json")

    # --------------------------------------------------------
    # Step 5) 매칭된 track들 global_id 할당
    # --------------------------------------------------------
    gid_manager.assign_global_ids(matches)

    # --------------------------------------------------------
    # Step 6) 매칭 실패 track들을 global ID로 자동 분리
    # --------------------------------------------------------
    cam0_tracks, cam1_tracks = get_all_tracks("track_embeddings")

    gid_manager.assign_unmatched_tracks(
        cam0_tracks=cam0_tracks,
        cam1_tracks=cam1_tracks,
    )

    # --------------------------------------------------------
    # Step 7) global_id_map.json 저장
    # --------------------------------------------------------
    gid_manager.save()


if __name__ == "__main__":
    main()
