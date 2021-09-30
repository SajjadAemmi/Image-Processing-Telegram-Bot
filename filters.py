import numpy as np
import cv2
from landmarks_detector import LandmarksDetector


def dodge(x,y):
    return cv2.divide(x, 255-y, scale=256)


def burn(image, mask):
    return 255 - cv2.divide(255-image, 255-mask, scale=256)


def image2pencilSketch(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_inv = 255 - image_gray
    image_gray_inv_blur = cv2.GaussianBlur(image_gray_inv, (21, 21), sigmaX=0, sigmaY=0)
    image_dodged = dodge(image_gray, image_gray_inv_blur)
    image_result = burn(image_dodged, image_gray_inv_blur)
    return image_result


def image2gray(image_path):
    image = cv2.imread(image_path)
    image_result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_result


def landmarks2image(image, background, landmarks):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [landmarks], -1, (255,255,255), -1)

    mask = cv2.GaussianBlur(mask,(49,49),0)
    mask = cv2.multiply(cv2.subtract(mask, 150), 2)

    r_min = np.min(landmarks[:,1])
    r_max = np.max(landmarks[:,1])
    c_min = np.min(landmarks[:,0])
    c_max = np.max(landmarks[:,0])
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

    # Convert uint8 to float
    foreground = lips_new.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = mask_new.astype(float)/255
    alpha = cv2.merge((alpha, alpha, alpha))

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    result = cv2.add(foreground, background)

    return result


def face_eyes_lips(image_path):
    image = cv2.imread(image_path)
    background = image.astype(float)
    landmarks_detector = LandmarksDetector()    
    all_face_landmarks = landmarks_detector.get_landmarks(image_path)

    for face_landmarks in all_face_landmarks:
        left_eye_landmarks = face_landmarks[36:42]
        background = landmarks2image(image, background, left_eye_landmarks)

        right_eye_landmarks = face_landmarks[42:48]
        background = landmarks2image(image, background, right_eye_landmarks)

        lips_landmarks = face_landmarks[48:68]
        background = landmarks2image(image, background, lips_landmarks)
        
    result = background.astype(np.uint8)
    return result
    


if __name__ == "__main__":
    result = face_eyes_lips("input/photos/file_0.jpg")
    cv2.imshow('output', result)
    cv2.waitKey(0)
