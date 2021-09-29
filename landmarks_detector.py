import dlib
import numpy as np
import cv2


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
        return all_face_landmarks


if __name__ == "__main__":
    landmarks_detector = LandmarksDetector()    
    all_face_landmarks = landmarks_detector.get_landmarks("input/photos/file_0.jpg")
    # print(all_face_landmarks)

    image = cv2.imread("input/photos/file_0.jpg")
    mask = np.zeros(image.shape[:2], np.uint8)
    
    for face_landmarks in all_face_landmarks:
        lips_landmarks = np.array(face_landmarks[48:68], dtype=np.int32)
        cv2.drawContours(mask, [lips_landmarks], 0, (255,255,255), -1)

        x_min = np.min(lips_landmarks[:,0])
        x_max = np.max(lips_landmarks[:,0])
        y_min = np.min(lips_landmarks[:,1])
        y_max = np.max(lips_landmarks[:,1])
        x_center = (x_max + x_min) // 2
        y_center = (y_max + y_min) // 2

        image_lips = image[y_min:y_max, x_min:x_max]
        mask_lips = mask[y_min:y_max, x_min:x_max]

        lips = cv2.bitwise_and(image_lips, image_lips, mask=mask_lips)
        lips = cv2.resize(lips, (0, 0), fx=3, fy=3)
        mask_lips = cv2.resize(mask_lips, (0, 0), fx=3, fy=3)

        lips_w, lips_h, _ = lips.shape
        lips_x_min = x_center - lips_w // 2
        lips_y_min = y_center - lips_h // 2

        mask_new = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_new[lips_x_min:lips_x_min + lips_w, lips_y_min:lips_y_min + lips_h] = mask_lips
        
        lips_new = np.zeros(image.shape, dtype=np.uint8)
        lips_new[lips_x_min:lips_x_min + lips_w, lips_y_min:lips_y_min + lips_h] = lips

        # Convert uint8 to float
        foreground = lips_new.astype(float)
        background = image.astype(float)
        
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = lips_new.astype(float)/255

        print(alpha.shape)
        print(foreground.shape)

        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)
        
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)
        
        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)
        

    cv2.imshow('output', lips_new)
    cv2.waitKey(0)
