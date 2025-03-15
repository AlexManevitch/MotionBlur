from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import cv2
import numpy as np

from consts import NUM_ARRAYS, ARRAY_SHAPE


class VideoStreamer:
    def __init__(self, video_path: str, shared_mem: SharedMemory = None):
        self.video_path: str = video_path
        self.continue_streaming: bool = False
        self._stream: cv2.VideoCapture = None
        self.all_frames = np.ndarray((NUM_ARRAYS, *ARRAY_SHAPE), dtype=np.float64, buffer=shared_mem.buf)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def start_streaming(self):
        self.continue_streaming = True
        self._stream = cv2.VideoCapture(self.video_path)

        if not self._stream.isOpened():
            print("Error: Unable to open video source.")
            exit()
        current_frame_count = 0
        while self.continue_streaming:
            self.continue_streaming, current_frame = self._stream.read()

            if not self.continue_streaming:
                break

            current_frame = cv2.resize(current_frame, (ARRAY_SHAPE[1], ARRAY_SHAPE[0]))
            self.all_frames[current_frame_count] = current_frame
            current_frame_count = (current_frame_count + 1) % self.all_frames.shape[0]

    def __del__(self):
        self._stream.release()
