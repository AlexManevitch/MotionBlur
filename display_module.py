from datetime import datetime

import cv2


class DisplayModule:
    def __init__(self):
        pass

    @staticmethod
    def display_contours(frame, contours):
        for current_contour in contours:
            x, y, w, h = current_contour
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            # show the frame and record if the user presses a key
            cv2.imshow("Feed", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

