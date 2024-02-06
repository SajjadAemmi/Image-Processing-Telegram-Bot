import os
import cv2
import numpy as np
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis


def preprocess_face_dataset(dataset_path):
        
    model = FaceAnalysis(providers=['CPUExecutionProvider'])
    model.prepare(ctx_id=0, det_size=(640, 640))

    face_dataset = []
    for gender in ['female', 'male']:
        for file_name in tqdm(os.listdir(os.path.join(dataset_path, gender))):
            image_path = os.path.join(dataset_path, gender, file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = model.get(image)
            if len(faces) == 0:
                message = "No face detected"
                return None, message
            elif len(faces) > 1:
                message = "More than one face detected"
                return None, message

            face = faces[0]
            embedding = face["embedding"]
            face_dataset.append({'image_path': image_path, 'embedding': embedding})

    np.save('face_dataset.npy', face_dataset)


if __name__ == "__main__":
    dataset_path = "datasets/faces"
    preprocess_face_dataset(dataset_path)
