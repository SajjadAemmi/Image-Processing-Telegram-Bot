import numpy as np
from src.TFLiteFaceDetector import UltraLightFaceDetecion
from src.TFLiteFaceAlignment import CoordinateAlignmentModel
from config import face_detection_weights_path, face_alignment_weights_path, face_detection_conf_threshold


class LandmarksDetector:
    def __init__(self):
        self.face_detection = UltraLightFaceDetecion(face_detection_weights_path,
                                                     conf_threshold=face_detection_conf_threshold)
        self.face_alignment = CoordinateAlignmentModel(face_alignment_weights_path)

    def get_landmarks(self, image):
        boxes, scores = self.face_detection.inference(image)

        all_face_landmarks = []
        for pred in self.face_alignment.get_landmarks(image, boxes):
            face_landmarks = [tuple(p) for p in np.round(pred).astype(np.int64)]
            all_face_landmarks.append(face_landmarks)

        return np.array(all_face_landmarks, dtype=np.int32)
