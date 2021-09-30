import dlib
import numpy as np


class LandmarksDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        all_face_landmarks = []
        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            all_face_landmarks.append(face_landmarks)

        return np.array(all_face_landmarks, dtype=np.int32)
