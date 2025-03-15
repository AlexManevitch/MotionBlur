from typing import Optional

import cv2

from detector import Detector


class VideoStreamer:
    def __init__(self, video_path: str, desired_size: Optional[tuple[int, int]] = (0, 0)):
        self.video_path: str = video_path
        self._continue_streaming: bool = False
        self._desired_size: tuple[int, int] = desired_size
        self._stream: cv2.VideoCapture = None
        self.all_frames = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def start_streaming(self):
        self._continue_streaming = True
        self._stream = cv2.VideoCapture(self.video_path)

        if not self._stream.isOpened():
            print("Error: Unable to open video source.")
            exit()
        first_frame = None
        dt = None
        while self._continue_streaming:
            self._continue_streaming, current_frame = self._stream.read()

            if not self._continue_streaming:
                break

            if self._desired_size[0] and self._desired_size[1]:
                current_frame = cv2.resize(current_frame, self._desired_size)
            self.all_frames.append(current_frame)


    def __del__(self):
        self._stream.release()
        cv2.destroyAllWindows()

