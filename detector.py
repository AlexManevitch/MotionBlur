from multiprocessing.shared_memory import SharedMemory

import cv2
import imutils
import datetime

import numpy as np

from consts import NUM_ARRAYS, ARRAY_SHAPE


class Detector:
    def __init__(self, background_image, minimum_contour_size=900, shared_mem:SharedMemory = None):
        self.background_image = background_image
        first_gray = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2GRAY)
        self.first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)
        self.minimum_contour_size = minimum_contour_size
        self.processed_images = np.ndarray((NUM_ARRAYS, *ARRAY_SHAPE), dtype=np.float64, buffer=shared_mem.buf)

        self.contours_per_image = np.ndarray((NUM_ARRAYS, 4), dtype=np.float64, buffer=shared_mem.buf)

    def _get_relevant_contours(self, frame):
        second_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.GaussianBlur(second_gray, (21, 21), 0)

        frame_delta = cv2.absdiff(self.first_gray, second_gray)
        thresh = cv2.threshold(frame_delta, 32, 255, cv2.ADAPTIVE_THRESH_MEAN_C)[1]
        thresh = cv2.dilate(thresh, None, iterations=4)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        return contours

    def _feed_relevant_contours_to_frame(self, contours, frame, frame_number) -> None:
        relevant_contours = [cv2.boundingRect(current_contour) for current_contour in
                             contours if cv2.contourArea(current_contour) > self.minimum_contour_size]
        self.processed_images[frame_number] = frame
        self.contours_per_image[frame_number] = relevant_contours

    def detect_image(self, frame, frame_number) -> None:
        contours = self._get_relevant_contours(frame=frame)
        self._feed_relevant_contours_to_frame(contours=contours, frame=frame, frame_number=frame_number)
