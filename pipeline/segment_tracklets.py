import os

import cv2
import numpy as np
from natsort import natsorted


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_sharpness(img):
    """
    이미지 선명도를 Laplacian variance 로 계산
    값이 클수록 더 선명한 이미지라고 볼 수 있음
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def segment_tracklet_and_select_best(
    input_base_dir="tracklets",
    output_base_dir="segments",
    segment_len=15,  # 30FPS 기준 0.5초
    min_frames_per_segment=3,
    image_exts=(".jpg", ".png", ".jpeg"),
):
    """
    tracklets/ 아래의 구조:
      tracklets/cam0/track_1/0.jpg, 1.jpg, ...
      tracklets/cam1/track_3/0.jpg, ...

    를 읽어서

    segments/cam0/track_1/seg_000_best.jpg
    segments/cam0/track_1/seg_001_best.jpg
    ...

    형태로 저장
    """
    for cam_name in os.listdir(input_base_dir):
        cam_path = os.path.join(input_base_dir, cam_name)
        if not os.path.isdir(cam_path):
            continue

        print(f"[INFO] Processing camera dir: {cam_path}")

        for track_name in os.listdir(cam_path):
            track_path = os.path.join(cam_path, track_name)
            if not os.path.isdir(track_path):
                continue

            print(f"  [TRACK] {cam_name}/{track_name}")

            # 프레임 파일들 정렬 (0.jpg, 1.jpg, 2.jpg ...)
            frame_files = [f for f in os.listdir(track_path) if f.lower().endswith(image_exts)]
            if not frame_files:
                continue

            frame_files = natsorted(frame_files)

            # output 디렉토리 준비
            out_track_dir = os.path.join(output_base_dir, cam_name, track_name)
            ensure_dir(out_track_dir)

            # segment 단위로 나누기
            num_frames = len(frame_files)
            seg_idx = 0

            for start in range(0, num_frames, segment_len):
                end = min(start + segment_len, num_frames)
                segment_files = frame_files[start:end]

                if len(segment_files) < min_frames_per_segment:
                    # 너무 짧은 segment 는 스킵 (원하면 저장해도 됨)
                    continue

                best_score = -1.0
                best_frame_file = None

                # segment 내에서 best frame 찾기
                for fname in segment_files:
                    fpath = os.path.join(track_path, fname)
                    img = cv2.imread(fpath)
                    if img is None:
                        continue

                    score = compute_sharpness(img)

                    if score > best_score:
                        best_score = score
                        best_frame_file = fpath

                if best_frame_file is None:
                    continue

                # best frame 저장
                seg_name = f"seg_{seg_idx:03d}_best.jpg"
                out_path = os.path.join(out_track_dir, seg_name)

                best_img = cv2.imread(best_frame_file)
                cv2.imwrite(out_path, best_img)

                print(
                    f"    [SEG {seg_idx:03d}] frames {start}~{end-1} -> {seg_name} (sharpness={best_score:.1f})"
                )

                seg_idx += 1


if __name__ == "__main__":
    # 예) 30FPS 기준 0.5초 단위 (15프레임씩 segment)
    segment_tracklet_and_select_best(
        input_base_dir="tracklets",
        output_base_dir="segments",
        segment_len=15,
        min_frames_per_segment=3,
    )
