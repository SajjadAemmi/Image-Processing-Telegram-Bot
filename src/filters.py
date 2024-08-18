import numpy as np
import cv2
from src.landmarks_detector import LandmarksDetector
from src.face_parts_detector import FacePartsDetector


def dodge(x, y):
    return cv2.divide(x, 255 - y, scale=256)


def burn(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)


def image2pencilSketch(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_inv = 255 - image_gray
    image_gray_inv_blur = cv2.GaussianBlur(image_gray_inv, (21, 21), sigmaX=0, sigmaY=0)
    image_dodged = dodge(image_gray, image_gray_inv_blur)
    image_result = burn(image_dodged, image_gray_inv_blur)
    return image_result


def image2gray(image):
    image_result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_result


def landmarks2image(image, background, landmarks):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [landmarks], -1, (255, 255, 255), -1)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = cv2.multiply(cv2.subtract(mask, 50), 2)

    r_min = np.min(landmarks[:, 1])
    r_max = np.max(landmarks[:, 1])
    c_min = np.min(landmarks[:, 0])
    c_max = np.max(landmarks[:, 0])
    r_center = (r_max + r_min) // 2
    c_center = (c_max + c_min) // 2

    image_lips = image[r_min:r_max, c_min:c_max]
    mask_lips = mask[r_min:r_max, c_min:c_max]

    lips = cv2.bitwise_and(image_lips, image_lips, mask=mask_lips)
    lips = cv2.resize(lips, (0, 0), fx=2, fy=2)
    mask_lips = cv2.resize(mask_lips, (0, 0), fx=2, fy=2)

    lips_w, lips_h, _ = lips.shape
    lips_x_min = r_center - lips_w // 2
    lips_y_min = c_center - lips_h // 2

    mask_new = np.zeros(image.shape[:2], dtype=np.uint8)
    mask_new[lips_x_min:lips_x_min + lips_w, lips_y_min:lips_y_min + lips_h] = mask_lips

    lips_new = np.zeros(image.shape, dtype=np.uint8)
    lips_new[lips_x_min:lips_x_min + lips_w, lips_y_min:lips_y_min + lips_h] = lips
    foreground = lips_new.astype(float)  # Convert uint8 to float

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = mask_new.astype(float) / 255
    alpha = cv2.merge((alpha, alpha, alpha))

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    result = cv2.add(foreground, background)

    return result


def face_eyes_lips(image_path):
    image = cv2.imread(image_path)
    background = image.astype(float)
    landmarks_detector = LandmarksDetector()
    all_face_landmarks = landmarks_detector.get_landmarks(image)
    lips_landmarks_indexes = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]
    left_eye_landmarks_indexes = [35, 41, 40, 42, 39, 37, 33, 36]
    right_eye_landmarks_indexes = [89, 95, 94, 96, 93, 91, 87, 90]

    for face_landmarks in all_face_landmarks:
        left_eye_landmarks = np.array([face_landmarks[i] for i in left_eye_landmarks_indexes])
        background = landmarks2image(image, background, left_eye_landmarks)

        right_eye_landmarks = np.array([face_landmarks[i] for i in right_eye_landmarks_indexes])
        background = landmarks2image(image, background, right_eye_landmarks)

        lips_landmarks = np.array([face_landmarks[i] for i in lips_landmarks_indexes])
        background = landmarks2image(image, background, lips_landmarks)

    result = background.astype(np.uint8)
    return result


def image2cartoon(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_blur = cv2.medianBlur(image_gray, 5)
    edges = cv2.adaptiveThreshold(image_gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    image_color = cv2.bilateralFilter(image, 9, 250, 250)
    image_result = cv2.bitwise_and(image_color, image_color, mask=edges)
    return image_result


def find_my_face(image, face_parts_detector):
    results = []

    result = face_parts_detector(image)
    results.append(result)

    # cv2.imwrite('output/aligned_image.jpg', aligned_image)
    # cv2.imwrite('output/left_eye_image.jpg', left_eye_image)
    # cv2.imwrite('output/right_eye_image.jpg', right_eye_image)
    # cv2.imwrite('output/nose_image.jpg', nose_image)
    # cv2.imwrite('output/lips_image.jpg', lips_image)

    face_dataset = np.load('face_dataset.npy', allow_pickle=True)
    face_embedding = [face['embedding'] for face in face_dataset]
    distances = np.linalg.norm(face_embedding - result["embedding"], axis=1)
    nearest_faces_indices = distances.argsort()[:3]

    for i in nearest_faces_indices:
        face = face_dataset[i]
        image_path = face['image_path']
        image = cv2.imread(image_path)
        result = face_parts_detector(image)
        results.append(result)

        # cv2.imwrite(f'output/aligned_image_{i}.jpg', result["aligned_image"])
        # cv2.imwrite(f'output/left_eye_image_{i}.jpg', result["left_eye_image"])
        # cv2.imwrite(f'output/right_eye_image_{i}.jpg', result["right_eye_image"])
        # cv2.imwrite(f'output/nose_image_{i}.jpg', result["nose_image"])
        # cv2.imwrite(f'output/lips_image_{i}.jpg', result["lips_image"])

    return results

if __name__ == "__main__":
    image_path = "input/photos/file_0.jpg"
    # result = image2cartoon(image_path)
    # result = image2gray(image_path)
    # result = image2pencilSketch(image_path)
    # result = face_eyes_lips(image_path)
    # cv2.imshow('output', result)
    # cv2.waitKey(0)

    image = cv2.imread(image_path)
    face_parts_detector = FacePartsDetector()
    find_my_face(image, face_parts_detector)
