import csv
import os
from dataclasses import dataclass


@dataclass
class TrackInfo:
    cam: str
    track: str
    start_time: float
    end_time: float
    cx_norm: float
    cy_norm: float


class TrackMetaManager:
    def __init__(self, video_fps=30):
        self.video_fps = video_fps

        # (cam, track_id) → dict
        self.track_info = {}

    def update_track(
        self,
        cam: str,
        track_id: int,
        frame_idx: int,
        cx_norm: float,
        cy_norm: float,
    ):
        key = (cam, track_id)

        # frame_idx 를 초로 변환
        time_sec = frame_idx / self.video_fps

        if key not in self.track_info:
            # 새로운 track 등장
            self.track_info[key] = TrackInfo(
                cam=cam,
                track=f"track_{track_id}",
                start_time=time_sec,
                end_time=time_sec,
                cx_norm=cx_norm,
                cy_norm=cy_norm,
            )
        else:
            # 기존 track 업데이트
            info = self.track_info[key]
            info.end_time = time_sec

            # bbox 중심은 최근 프레임 기준으로 저장(나중에 평균하도록 확장 가능)
            info.cx_norm = cx_norm
            info.cy_norm = cy_norm

    def save_csv(self, out_path="track_meta.csv"):
        # 디렉토리 경로가 있을 때만 디렉토리 생성
        dir_path = os.path.dirname(out_path)
        if dir_path:  # 빈 문자열이 아닐 때만
            os.makedirs(dir_path, exist_ok=True)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cam", "track", "start_time", "end_time", "cx_norm", "cy_norm"])

            for (cam, track_id), info in self.track_info.items():
                writer.writerow(
                    [
                        info.cam,
                        info.track,
                        f"{info.start_time:.3f}",
                        f"{info.end_time:.3f}",
                        f"{info.cx_norm:.5f}",
                        f"{info.cy_norm:.5f}",
                    ]
                )

        print(f"[TrackMeta] Saved metadata: {out_path}")
