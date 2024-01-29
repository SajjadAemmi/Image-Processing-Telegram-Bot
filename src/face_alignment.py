import cv2
import numpy as np
from skimage import transform


class FaceAlignment:
    def __init__(self):
        self.reference_landmarks = np.array(
            [[38.2946, 51.6963],  # left eye center
            [73.5318, 51.5014],  # right eye center
            [56.0252, 71.7366],  # nose tip
            [41.5493, 92.3655],  # mouth left corner
            [70.7299, 92.2041]], # mouth right corner
            dtype=np.float32)

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
    
    def preprocess(self, input_image):
        output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        return output_image

    def __call__(self, input_image, face_landmarks):
        output_image, M = self.norm_crop(input_image, face_landmarks, 112)
        return output_image, M
