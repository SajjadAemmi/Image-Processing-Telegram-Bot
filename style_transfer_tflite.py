import cv2
import numpy as np
import tensorflow as tf
from config import style_predict_weights_path, style_transform_weights_path

# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
    img = cv2.imread(path_to_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def resize_with_crop_or_pad(image, target_w, target_h):
    cols, rows = image.shape[:2]
    center_r = rows // 2
    center_c = cols // 2

    # crop
    if rows > target_h:
        r1 = center_r - target_h // 2
        r2 = center_r + target_h // 2
    else:
        r1 = 0
        r2 = rows

    if cols > target_w:
        c1 = center_c - target_w // 2
        c2 = center_c + target_w // 2
    else:
        c1 = 0
        c2 = cols

    image = image[c1:c2, r1:r2]

    # pad
    if rows < target_w:
        v_border = (target_w - rows) // 2
    else:
        v_border = 0

    if cols < target_h:
        h_border = (target_h - cols) // 2
    else:
        h_border = 0

    white = [255, 255, 255]
    image = cv2.copyMakeBorder(image, v_border, v_border, h_border, h_border, cv2.BORDER_CONSTANT, value=white)

    return image


# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
    shape = image.shape[:2]
    short_dim = min(shape)
    scale = target_dim / short_dim

    new_h = int(shape[0] * scale)
    new_w = int(shape[1] * scale)

    image = cv2.resize(image, (new_w, new_h))
    # Central crop the image.
    image = resize_with_crop_or_pad(image, target_dim, target_dim)
    image = np.expand_dims(image, axis=0)
    return image


def imshow(image, title='output'):
    if len(image.shape) > 3:
        image = np.squeeze(image, axis=0)

    image = image * 255
    image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output/photos/result.jpg', image)
    cv2.imshow(title, image)
    cv2.waitKey()


# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_predict_weights_path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    return style_bottleneck


# Run style transform on preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_transform_weights_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image.
    stylized_image = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    return stylized_image


def styleTransfer(content_path, style_path):
    # Load the input images.
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Preprocess the input images.
    preprocessed_content_image = preprocess_image(content_image, 384)
    preprocessed_style_image = preprocess_image(style_image, 256)

    print('Style Image Shape:', preprocessed_style_image.shape)
    print('Content Image Shape:', preprocessed_content_image.shape)

    # imshow(preprocessed_content_image, 'Content Image')
    # imshow(preprocessed_style_image, 'Style Image')

    # Calculate style bottleneck for the preprocessed style image.
    style_bottleneck = run_style_predict(preprocessed_style_image)
    print('Style Bottleneck Shape:', style_bottleneck.shape)

    """### Style transform"""
    # Stylize the content image using the style bottleneck.
    stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image)

    # imshow(stylized_image, 'Stylized Image')

    # Calculate style bottleneck of the content image.
    style_bottleneck_content = run_style_predict(preprocess_image(content_image, 256))

    # Define content blending ratio between [0..1].
    # 0.0: 0% style extracts from content image.
    # 1.0: 100% style extracted from content image.
    content_blending_ratio = 0.5

    # Blend the style bottleneck of style image and content image
    style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (
                1 - content_blending_ratio) * style_bottleneck

    # Stylize the content image using the style bottleneck.
    stylized_image_blended = run_style_transform(style_bottleneck_blended, preprocessed_content_image)

    # imshow(stylized_image_blended, 'Blended Stylized Image')

    result = stylized_image_blended

    if len(result.shape) > 3:
        result = np.squeeze(result, axis=0)

    result = result * 255
    result = result.astype(np.uint8)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


if __name__ == "__main__":
    content_path = "input/photos/file_0.jpg"
    style_path = "input/styles/TheScream-EdvardMunch.jpg"

    image = styleTransfer(content_path, style_path)
    cv2.imshow('output', image)
    cv2.waitKey()
