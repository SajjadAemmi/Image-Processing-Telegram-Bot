import tensorflow_hub as hub
import numpy as np
import PIL.Image
import tensorflow as tf



def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

content_image = load_img(content_path)
style_image = load_img(style_path)
stylized_image = hub_model(tf.constant(content_image), tf.constant(my_style_image))[0]
tensor_to_image(stylized_image)
