import cv2
import imutils
import datetime


class Detector:
    def __init__(self, background_image, minimum_contour_size=900):
        self.background_image = background_image
        first_gray = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2GRAY)
        self.first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)
        self.minimum_contour_size = minimum_contour_size
        self.processed_images = []
        self.contours_per_image = []

    def detect_image(self, frame):
        second_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.GaussianBlur(second_gray, (21, 21), 0)

        frame_delta = cv2.absdiff(self.first_gray, second_gray)
        thresh = cv2.threshold(frame_delta, 32, 255, cv2.ADAPTIVE_THRESH_MEAN_C)[1]
        thresh = cv2.dilate(thresh, None, iterations=4)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # getting only large enough contours ( to improve
        relevant_contours = [cv2.boundingRect(current_contour) for current_contour in
                             contours if cv2.contourArea(current_contour) < 900]
        self.processed_images.append(frame)
        self.contours_per_image.append(relevant_contours)
