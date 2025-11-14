import os
import queue
import threading
import time
from collections import deque
from urllib.parse import unquote

import cv2
from ultralytics import YOLO

# =============================================================
# Tracklet 저장 매니저 (비동기 I/O 지원)
# =============================================================


class TrackletManager:
    def __init__(self, base_dir="tracklets", async_save=True, queue_size=100):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.async_save = async_save

        if async_save:
            self.save_queue = queue.Queue(maxsize=queue_size)
            self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
            self.save_thread.start()

    def _save_worker(self):
        """비동기 저장 워커 스레드"""
        while True:
            try:
                cam_id, track_id, frame_idx, crop_img = self.save_queue.get(timeout=1.0)
                if cam_id is None:  # 종료 신호
                    break

                tracklet_dir = f"{self.base_dir}/cam{cam_id}/track_{track_id}"
                os.makedirs(tracklet_dir, exist_ok=True)
                file_path = f"{tracklet_dir}/{frame_idx}.jpg"
                cv2.imwrite(file_path, crop_img)
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving crop: {e}")
                self.save_queue.task_done()

    def save_crop(self, cam_id, track_id, frame_idx, crop_img):
        """
        Save cropped person image into:
        tracklets/cam{cam_id}/track_{track_id}/{frame_idx}.jpg
        """
        if self.async_save:
            try:
                self.save_queue.put_nowait((cam_id, track_id, frame_idx, crop_img))
            except queue.Full:
                # 큐가 가득 찬 경우 동기적으로 저장 (드롭 방지)
                tracklet_dir = f"{self.base_dir}/cam{cam_id}/track_{track_id}"
                os.makedirs(tracklet_dir, exist_ok=True)
                file_path = f"{tracklet_dir}/{frame_idx}.jpg"
                cv2.imwrite(file_path, crop_img)
        else:
            tracklet_dir = f"{self.base_dir}/cam{cam_id}/track_{track_id}"
            os.makedirs(tracklet_dir, exist_ok=True)
            file_path = f"{tracklet_dir}/{frame_idx}.jpg"
            cv2.imwrite(file_path, crop_img)

    def close(self):
        """비동기 저장 완료 대기"""
        if self.async_save:
            self.save_queue.put((None, None, None, None))  # 종료 신호
            self.save_queue.join()  # 모든 작업 완료 대기


# =============================================================
# 카메라별 YOLO Tracking 실행 함수
# =============================================================


def camera_worker(
    cam_id, rtsp_url, yolo_model, tracklet_manager, show_preview=False, max_retries=3
):
    """
    cam_id: (0 or 1)
    rtsp_url: RTSP stream URL
    yolo_model: YOLO model instance
    tracklet_manager: TrackletManager instance
    show_preview: Whether to show preview window (requires display)
    max_retries: Maximum number of connection retry attempts
    """
    # URL 디코딩 처리 (%21 -> !)
    rtsp_url_decoded = unquote(rtsp_url)
    print(f"[Cam {cam_id}] Starting RTSP stream: {rtsp_url_decoded}")

    cap = None
    # 카메라별로 연결 시도 간격을 두기 위해 초기 지연
    time.sleep(cam_id * 0.5)

    for attempt in range(max_retries):
        try:
            print(
                f"[Cam {cam_id}] Attempt {attempt + 1}/{max_retries}: "
                "Trying RTSP connection with default backend..."
            )

            # 기본 백엔드를 먼저 시도 (Cam 1이 성공한 방법)
            cap = cv2.VideoCapture(rtsp_url_decoded)

            # RTSP 연결 옵션 설정
            # 버퍼 크기 조정: 너무 작으면 프레임 드롭, 너무 크면 지연 발생
            # 3으로 설정하여 프레임 드롭을 줄이면서도 지연을 최소화
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            # 타임아웃 설정 (밀리초)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            # 연결 확인을 위해 충분한 대기 후 프레임 읽기 시도
            time.sleep(1.0)
            ret, frame = cap.read()

            if cap.isOpened() and ret and frame is not None:
                print(f"[Cam {cam_id}] Successfully connected to RTSP stream")
                break
            else:
                if cap:
                    cap.release()
                cap = None
                print(f"[Cam {cam_id}] Connection attempt {attempt + 1} failed")

                # TCP 전송 프로토콜을 사용한 재시도
                if attempt < max_retries - 1:
                    print(f"[Cam {cam_id}] Trying with TCP transport...")
                    rtsp_options = (
                        f"{rtsp_url_decoded}?rtsp_transport=tcp"
                        if "?" not in rtsp_url_decoded
                        else f"{rtsp_url_decoded}&rtsp_transport=tcp"
                    )
                    cap = cv2.VideoCapture(rtsp_options)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                    time.sleep(1.0)
                    ret, frame = cap.read()
                    if cap.isOpened() and ret and frame is not None:
                        print(f"[Cam {cam_id}] Successfully connected with TCP transport")
                        break
                    else:
                        if cap:
                            cap.release()
                        cap = None

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"[Cam {cam_id}] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

        except Exception as e:
            print(f"[Cam {cam_id}] Exception during connection attempt {attempt + 1}: {e}")
            if cap:
                cap.release()
                cap = None
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)

    if cap is None or not cap.isOpened():
        print(f"[Cam {cam_id}] ERROR: Failed to open RTSP stream after {max_retries} attempts")
        return

    frame_idx = 0
    consecutive_failures = 0
    max_consecutive_failures = 30  # 30번 연속 실패 시 재연결 시도

    # 프레임 버퍼: 최신 프레임만 유지하여 프레임 드롭 시에도 최신 프레임 사용
    frame_buffer = deque(maxlen=1)
    last_valid_frame = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"[Cam {cam_id}] Too many consecutive failures, attempting reconnection...")
                cap.release()
                time.sleep(2)
                # 재연결 시도 (기본 백엔드 사용)
                cap = cv2.VideoCapture(rtsp_url_decoded)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                time.sleep(1.5)
                ret, frame = cap.read()
                if not cap.isOpened() or not ret or frame is None:
                    print(f"[Cam {cam_id}] Reconnection failed, terminating worker.")
                    break
                consecutive_failures = 0
                frame_buffer.clear()  # 재연결 시 버퍼 초기화
                last_valid_frame = None
                print(f"[Cam {cam_id}] Reconnected successfully")
            else:
                # 프레임 읽기 실패 시 마지막 유효한 프레임 사용 (tracking 연속성 유지)
                if last_valid_frame is not None:
                    frame = last_valid_frame.copy()
                    ret = True
                else:
                    time.sleep(0.1)  # 짧은 대기 후 재시도
                    continue
            if not ret:
                continue

        consecutive_failures = 0  # 성공 시 카운터 리셋
        last_valid_frame = frame.copy()  # 유효한 프레임 저장
        frame_buffer.append(frame)

        # YOLO tracking with ByteTrack
        # persist=True로 tracking 상태 유지 (같은 사람이 같은 ID 유지)
        # conf와 iou를 조정하여 detection 성능 최적화
        results = yolo_model.track(
            frame,
            persist=True,  # tracking 상태 유지 (중요!)
            verbose=False,
            tracker="bytetrack.yaml",
            conf=0.25,  # detection confidence threshold (낮춰서 더 많은 detection)
            iou=0.45,  # NMS IoU threshold
        )

        # Parse tracking results
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0:  # class 0 = person
                    continue

                # track id (must exist when tracking = True)
                if box.id is None:
                    continue

                track_id = int(box.id[0])

                # bbox coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # crop person image
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Save crop
                tracklet_manager.save_crop(
                    cam_id=cam_id, track_id=track_id, frame_idx=frame_idx, crop_img=crop
                )

                # Optional: visualize tracking bbox (for debugging)
                if show_preview:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID:{track_id}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

        # Debug preview window (only if display is available)
        if show_preview:
            cv2.imshow(f"Camera {cam_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    print(f"[Cam {cam_id}] RTSP worker terminated.")


# =============================================================
# Main function: run two RTSP streams in parallel
# =============================================================


def main():
    # 각 카메라별로 별도의 YOLO 모델 인스턴스 생성 (thread-safe)
    # 같은 모델 인스턴스를 공유하면 tracking 상태가 충돌할 수 있음
    model_cam0 = YOLO("yolo11n.pt")
    model_cam1 = YOLO("yolo11n.pt")

    # Tracklet manager (비동기 저장 활성화)
    manager = TrackletManager("tracklets", async_save=True, queue_size=100)

    # Replace with your actual RTSP URLs
    rtsp_cam0 = "rtsp://admin:cubox2024%21@172.16.150.130:554/onvif/media?profile=M1_Profile1"
    rtsp_cam1 = "rtsp://admin:cubox2024%21@172.16.150.129:554/onvif/media?profile=M1_Profile1"

    # 서버 환경에서는 GUI 미사용 (DISPLAY 환경변수 확인)
    show_preview = os.environ.get("DISPLAY") is not None

    # Thread for each camera (각각 별도의 모델 인스턴스 사용)
    t0 = threading.Thread(
        target=camera_worker, args=(0, rtsp_cam0, model_cam0, manager, show_preview)
    )
    t1 = threading.Thread(
        target=camera_worker, args=(1, rtsp_cam1, model_cam1, manager, show_preview)
    )

    # Start
    t0.start()
    t1.start()

    try:
        # Wait for finish
        t0.join()
        t1.join()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user, shutting down...")
    finally:
        # 비동기 저장 완료 대기
        manager.close()
        print("[Main] All workers terminated.")


if __name__ == "__main__":
    main()
