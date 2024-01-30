import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


class FacePartsDetector:
    """ Detects face parts (left eye, right eye, nose, lips) from an image. """
    def __init__(self) -> None:
        self.model = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))
        self.left_eye_landmarks_indices = [35, 41, 40, 42, 39, 37, 33, 36]
        self.right_eye_landmarks_indices = [89, 95, 94, 96, 93, 91, 87, 90]
        self.nose_landmarks_indices = [86, 85, 84, 83, 82, 81, 80, 79, 78, 77]
        self.lips_landmarks_indices = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]

    def crop_image_with_landmarks(self, image, landmarks):
        """ Crop image with landmarks. """
        width, height = image.shape[:2]
        r_min = np.min(landmarks[:, 1]) - height // 40
        r_max = np.max(landmarks[:, 1]) + height // 40
        c_min = np.min(landmarks[:, 0]) - width // 40
        c_max = np.max(landmarks[:, 0]) + width // 40
        cropped_image = image[r_min:r_max, c_min:c_max]
        return cropped_image
    
    def face_parts(self, image, face_landmarks):
        """ Crop face parts from image. """
        left_eye_landmarks = np.take(face_landmarks, self.left_eye_landmarks_indices, axis=0)
        left_eye_image = self.crop_image_with_landmarks(image, left_eye_landmarks)

        right_eye_landmarks = np.take(face_landmarks, self.right_eye_landmarks_indices, axis=0)
        right_eye_image = self.crop_image_with_landmarks(image, right_eye_landmarks)

        nose_landmarks = np.take(face_landmarks, self.nose_landmarks_indices, axis=0)
        nose_image = self.crop_image_with_landmarks(image, nose_landmarks)

        lips_landmarks = np.take(face_landmarks, self.lips_landmarks_indices, axis=0)
        lips_image = self.crop_image_with_landmarks(image, lips_landmarks)

        return left_eye_image, right_eye_image, nose_image, lips_image

    def __call__(self, input_image):
        faces = self.model.get(input_image)
        if len(faces) == 0:
            message = "No face detected"
            return None, message
        elif len(faces) > 1:
            message = "More than one face detected"
            return None, message

        face = faces[0]
        embedding = face["embedding"]
        aligned_image, M = insightface.utils.face_align.norm_crop2(input_image, face["kps"], 112)
        landmark_2d_106 = face["landmark_2d_106"].reshape(1, 106, 2)
        transformed_landmark_2d_106 = cv2.transform(landmark_2d_106, M)
        transformed_landmark_2d_106 = np.squeeze(transformed_landmark_2d_106).astype(int)
        left_eye_image, right_eye_image, nose_image, lips_image = self.face_parts(aligned_image, transformed_landmark_2d_106)

        return aligned_image, left_eye_image, right_eye_image, nose_image, lips_image


if __name__ == "__main__":
    image_path = "input/photos/IMG_4670.JPG"
    image = cv2.imread(image_path)

    face_parts_detector = FacePartsDetector()
    aligned_image, left_eye_image, right_eye_image, nose_image, lips_image = face_parts_detector(image)

    cv2.imwrite('output/aligned_image.jpg', aligned_image)
    cv2.imwrite('output/left_eye_image.jpg', left_eye_image)
    cv2.imwrite('output/right_eye_image.jpg', right_eye_image)
    cv2.imwrite('output/nose_image.jpg', nose_image)
    cv2.imwrite('output/lips_image.jpg', lips_image)
