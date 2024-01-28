import numpy as np
import cv2
from src.landmarks_detector import LandmarksDetector
from src.face_alignment import FaceAlignment


class FacePartsDetector:
    """ Detects face parts (left eye, right eye, nose, lips) in an image """
    def __init__(self):
        self.landmarks_detector = LandmarksDetector()
        self.face_alignment = FaceAlignment()

    def crop_image_with_landmarks(self, image, landmarks):
        r_min = np.min(landmarks[:, 1])
        r_max = np.max(landmarks[:, 1])
        c_min = np.min(landmarks[:, 0])
        c_max = np.max(landmarks[:, 0])
        cropped_image = image[r_min:r_max, c_min:c_max]
        return cropped_image

    def preprocess(self, input_image):
        output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        return output_image

    def postprocess(self, image, face_landmarks):
        left_eye_landmarks_indices = [35, 41, 40, 42, 39, 37, 33, 36]
        right_eye_landmarks_indices = [89, 95, 94, 96, 93, 91, 87, 90]
        nose_landmarks_indices = [86, 85, 84, 83, 82, 81, 80, 79, 78, 77]
        lips_landmarks_indices = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]

        left_eye_landmarks = np.take(face_landmarks, left_eye_landmarks_indices, axis=0)
        left_eye_image = self.crop_image_with_landmarks(image, left_eye_landmarks)

        right_eye_landmarks = np.take(face_landmarks, right_eye_landmarks_indices, axis=0)
        right_eye_image = self.crop_image_with_landmarks(image, right_eye_landmarks)

        nose_landmarks = np.take(face_landmarks, nose_landmarks_indices, axis=0)
        nose_image = self.crop_image_with_landmarks(image, nose_landmarks)

        lips_landmarks = np.take(face_landmarks, lips_landmarks_indices, axis=0)
        lips_image = self.crop_image_with_landmarks(image, lips_landmarks)

        return left_eye_image, right_eye_image, nose_image, lips_image

    def __call__(self, input_image):
        output_image = input_image.copy()
        input_image = self.preprocess(input_image)
        all_face_landmarks = self.landmarks_detector.get_landmarks(input_image)

        if len(all_face_landmarks) == 0:
            message = "No face detected"
            return None, message
        elif len(all_face_landmarks) > 1:
            message = "More than one face detected"
            return None, message

        indices = [
            34,  # left eye center
            88,  # right eye center
            86,  # nose tip
            52,  # mouth left corner
            61,  # mouth right corner
            ]
        face_landmarks = np.take(all_face_landmarks[0], indices, axis=0)
        output_image, M = self.face_alignment(output_image, face_landmarks)
        transformed_face_landmarks = cv2.transform(all_face_landmarks, M)[0]
        left_eye_image, right_eye_image, nose_image, lips_image = self.postprocess(output_image, transformed_face_landmarks)

        return left_eye_image, right_eye_image, nose_image, lips_image


if __name__ == '__main__':
    image_path = "input/photos/IMG_4670.JPG"
    image = cv2.imread(image_path)

    DEFAULT_CROP_SIZE = (96, 112)

    face_parts_detector = FacePartsDetector()
    left_eye_image, right_eye_image, nose_image, lips_image = face_parts_detector(image)

    # Display the original and aligned faces
    cv2.imshow('Left Eye', left_eye_image)
    cv2.imshow('Right Eye', right_eye_image)
    cv2.imshow('Nose', nose_image)
    cv2.imshow('Lips', lips_image)
    cv2.waitKey(0)
    # cv2.imwrite('aligned_face.jpg', aligned_face)
    # cv2.destroyAllWindows()
