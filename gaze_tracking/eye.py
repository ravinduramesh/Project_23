import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _isolate(self, frame, landmarks, points):
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)

        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        margin = 5

        self.frame = eye[(np.min(region[:, 1]) - margin):(np.max(region[:, 1]) + margin), (np.min(region[:, 0]) - margin):(np.max(region[:, 0]) + margin)]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        
        top = (int((landmarks.part(points[1]).x + landmarks.part(points[2]).x) / 2), int((landmarks.part(points[1]).y + landmarks.part(points[2]).y) / 2))
        bottom = (int((landmarks.part(points[5]).x + landmarks.part(points[4]).x) / 2), int((landmarks.part(points[5]).y + landmarks.part(points[4]).y) / 2))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        if side == 0:
            points = [36, 37, 38, 39, 40, 41]
        elif side == 1:
            points = [42, 43, 44, 45, 46, 47]
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
