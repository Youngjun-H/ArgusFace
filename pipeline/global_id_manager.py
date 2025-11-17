import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class MatchCandidate:
    cam0_track: str
    cam1_track: str
    sim: float
    dt: float
    dx: float
    dy: float


class GlobalIDManager:
    def __init__(self, save_path="global_id_map.json"):
        """
        Global ID 관리 클래스

        global_map 구조:
        {
            "global_0001": {
                "cam0": ["track_3"],
                "cam1": ["track_12"]
            }
        }
        """
        self.global_map: Dict[str, Dict[str, List[str]]] = {}
        self.used_cam0 = set()
        self.used_cam1 = set()
        self.next_id = 1
        self.save_path = save_path

    # ---------------------------------------------------------
    # 내부 함수: 새로운 global_id 생성
    # ---------------------------------------------------------
    def _new_global_id(self) -> str:
        gid = f"global_{self.next_id:04d}"
        self.next_id += 1
        self.global_map[gid] = {"cam0": [], "cam1": []}
        return gid

    # ---------------------------------------------------------
    # track을 global ID 그룹에 추가
    # ---------------------------------------------------------
    def _assign_to_global(self, gid: str, cam: str, track: str):
        self.global_map[gid][cam].append(track)

        if cam == "cam0":
            self.used_cam0.add(track)
        else:
            self.used_cam1.add(track)

    # ---------------------------------------------------------
    # 주 기능: 매칭 리스트로 global ID 할당
    # ---------------------------------------------------------
    def assign_global_ids(self, matches: List[MatchCandidate]):
        """
        matches: similarity 내림차순 정렬된 MatchCandidate 리스트
        """

        for m in matches:
            c0 = m.cam0_track
            c1 = m.cam1_track

            # 이미 다른 global id에 사용된 track이면 skip
            if c0 in self.used_cam0 or c1 in self.used_cam1:
                continue

            # 새로운 global_id 생성
            gid = self._new_global_id()

            # 두 트랙을 global_id에 배정
            self._assign_to_global(gid, "cam0", c0)
            self._assign_to_global(gid, "cam1", c1)

            print(f"[GlobalID] {gid} ← cam0/{c0} ↔ cam1/{c1} " f"(sim={m.sim:.3f}, dt={m.dt:.2f}, dx={m.dx:.3f})")

    # ---------------------------------------------------------
    # meta 없는 나머지 track 처리 (옵션)
    # ---------------------------------------------------------
    def assign_unmatched_tracks(self, cam0_tracks: List[str], cam1_tracks: List[str]):
        """
        cross-camera match에 실패한 단독 track들도 global_id 생성

        이유:
        - 사람 두 카메라에 모두 잡히지 않을 수 있음
        - 짧은 track, occlusion 등으로 매칭 실패 가능
        """

        # cam0 잔여
        for t in cam0_tracks:
            if t not in self.used_cam0:
                gid = self._new_global_id()
                self._assign_to_global(gid, "cam0", t)
                print(f"[GlobalID][single] {gid} ← cam0/{t}")

        # cam1 잔여
        for t in cam1_tracks:
            if t not in self.used_cam1:
                gid = self._new_global_id()
                self._assign_to_global(gid, "cam1", t)
                print(f"[GlobalID][single] {gid} ← cam1/{t}")

    # ---------------------------------------------------------
    # 저장
    # ---------------------------------------------------------
    def save(self):
        # 디렉토리 경로가 있을 때만 디렉토리 생성
        dir_path = os.path.dirname(self.save_path)
        if dir_path:  # 빈 문자열이 아닐 때만
            os.makedirs(dir_path, exist_ok=True)

        with open(self.save_path, "w") as f:
            json.dump(self.global_map, f, indent=2)

        print(f"[GlobalID] Saved: {self.save_path}")
