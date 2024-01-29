import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from skimage import transform
from src.face_parts_detector import FacePartsDetector


class FindMe:
    def __init__(self) -> None:
        self.model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))
        self.reference_landmarks = np.array(
            [[38.2946, 51.6963],  # left eye center
            [73.5318, 51.5014],  # right eye center
            [56.0252, 71.7366],  # nose tip
            [41.5493, 92.3655],  # mouth left corner
            [70.7299, 92.2041]], # mouth right corner
            dtype=np.float32)
        
        self.face_parts_detector = FacePartsDetector()

    def estimate_norm(self, lmk, image_size=112):
        assert lmk.shape == (5, 2)
        assert image_size % 112 == 0 or image_size % 128 == 0
        if image_size % 112 == 0:
            ratio = float(image_size)/112.0
            diff_x = 0
        else:
            ratio = float(image_size)/128.0
            diff_x = 8.0*ratio
        dst = self.reference_landmarks * ratio
        dst[:, 0] += diff_x
        tform = transform.SimilarityTransform()
        tform.estimate(lmk, dst)
        M = tform.params[0:2, :]
        return M

    def norm_crop(self, img, landmark, image_size=112):
        M = self.estimate_norm(landmark, image_size)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped, M

    def __call__(self, input_image):
        faces = self.model.get(input_image)
        if len(faces) == 0:
            message = "No face detected"
            return None, message
        elif len(faces) > 1:
            message = "More than one face detected"
            return None, message

        face = faces[0]
        print(face)
        aligned_image, M = self.norm_crop(input_image, face["kps"], 112)
        landmark_2d_106 = face["landmark_2d_106"].reshape(1, 106, 2)
        transformed_landmark_2d_106 = cv2.transform(landmark_2d_106, M)
        transformed_landmark_2d_106 = np.squeeze(transformed_landmark_2d_106)
        left_eye_image, right_eye_image, nose_image, lips_image = self.face_parts_detector(aligned_image, transformed_landmark_2d_106)

        return aligned_image, left_eye_image, right_eye_image, nose_image, lips_image


if __name__ == "__main__":
    image_path = "input/photos/IMG_4670.JPG"
    image = cv2.imread(image_path)

    find_me = FindMe()
    aligned_image, left_eye_image, right_eye_image, nose_image, lips_image = find_me(image)

    cv2.imshow('Output Image', aligned_image)
    cv2.imshow('Left Eye Image', left_eye_image)
    cv2.imshow('Right Eye Image', right_eye_image)
    cv2.imshow('Nose Image', nose_image)
    cv2.imshow('Lips Image', lips_image)
    cv2.waitKey(0)