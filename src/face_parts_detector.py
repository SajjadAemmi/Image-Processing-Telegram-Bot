import numpy as np
import cv2


class FacePartsDetector:
    """ Detects face parts (left eye, right eye, nose, lips) in an image """
    def __init__(self):
        self.left_eye_landmarks_indices = [35, 41, 40, 42, 39, 37, 33, 36]
        self.right_eye_landmarks_indices = [89, 95, 94, 96, 93, 91, 87, 90]
        self.nose_landmarks_indices = [86, 85, 84, 83, 82, 81, 80, 79, 78, 77]
        self.lips_landmarks_indices = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]

    def crop_image_with_landmarks(self, image, landmarks):
        r_min = np.min(landmarks[:, 1])
        r_max = np.max(landmarks[:, 1])
        c_min = np.min(landmarks[:, 0])
        c_max = np.max(landmarks[:, 0])
        cropped_image = image[r_min:r_max, c_min:c_max]
        return cropped_image

    def __call__(self, image, face_landmarks):
        left_eye_landmarks = np.take(face_landmarks, self.left_eye_landmarks_indices, axis=0)
        print(left_eye_landmarks)
        left_eye_image = self.crop_image_with_landmarks(image, left_eye_landmarks)

        right_eye_landmarks = np.take(face_landmarks, self.right_eye_landmarks_indices, axis=0)
        right_eye_image = self.crop_image_with_landmarks(image, right_eye_landmarks)

        nose_landmarks = np.take(face_landmarks, self.nose_landmarks_indices, axis=0)
        nose_image = self.crop_image_with_landmarks(image, nose_landmarks)

        lips_landmarks = np.take(face_landmarks, self.lips_landmarks_indices, axis=0)
        lips_image = self.crop_image_with_landmarks(image, lips_landmarks)

        return left_eye_image, right_eye_image, nose_image, lips_image
