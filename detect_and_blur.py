import time

import cv2
import numpy as np

from consts import NUM_ARRAYS, ARRAY_SHAPE
from detector import Detector
from display_module import DisplayModule
from video_streamer import VideoStreamer
from multiprocessing import Process, Value, Array, shared_memory, freeze_support, Event

video_source = "data\\people_entering.mp4"

total_shape = (NUM_ARRAYS, *ARRAY_SHAPE)
#
# # Allocate shared memory
shared_mem = shared_memory.SharedMemory(create=True, size=int(np.prod(total_shape) * 8 * 3))  # 8 bytes for float64
start_event = Event()
start_event2 = Event()


def detector_worker():
    start_event.wait()
    count = 0
    start_event2.set()
    while True:
        detector.detect_image(vs.all_frames[count], count)
        count = (count + 1) % NUM_ARRAYS


def display_worker():
    for curr_frame, contours in zip(detector.processed_images, detector.contours_per_image):
        display_module.display_contours(curr_frame, contours)


def videostreamer_worker():
    vs.start_streaming()
    start_event.set()


vs = VideoStreamer(video_path=video_source, shared_mem=shared_mem)
detector = Detector(minimum_contour_size=900)
display_module = DisplayModule()
p1 = Process(target=vs.start_streaming, args=())
p2 = Process(target=detector_worker, args=[vs])
p3 = Process(target=videostreamer_worker, args=[vs])
processes = []
for p in [p1, p2, p3]:
    processes.append(p)
    p.start()
for p in processes:
    p.join()
    p.start()
