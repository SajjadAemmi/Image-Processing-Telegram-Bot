import os
from io import BytesIO
import cv2
import numpy as np
import qrcode
import telebot
from telebot import types
from src import filters

# from style_transfer import styleTransfer
import config
from dotenv import load_dotenv
from src.face_parts_detector import FacePartsDetector


load_dotenv()
Token = os.getenv("Token")
bot = telebot.TeleBot(Token)

markup = types.ReplyKeyboardMarkup(row_width=4)
itembtn1 = types.KeyboardButton("+")
itembtn2 = types.KeyboardButton("-")
itembtn3 = types.KeyboardButton("*")
itembtn4 = types.KeyboardButton("/")
markup.add(itembtn1, itembtn2, itembtn3, itembtn4)

bot_state = None
content_image_path = None

user_data = {}

face_parts_detector = FacePartsDetector()


@bot.message_handler(commands=["start"])
def send_start(message):
    bot.send_message(message.chat.id, config.start_test)


@bot.message_handler(commands=["help"])
def send_help(message):
    bot.send_message(message.chat.id, config.help_text)


@bot.message_handler(commands=["pencil_sketch"])
def pencil_sketch_image(message):
    bot.send_message(message.chat.id, "Send me an image")


@bot.message_handler(content_types=["photo"])
def pencil_sketch_image_next_step(message):
    image = message_photo_to_image(message)
    image_result = filters.image2pencilSketch(image)
    photo = image_to_message_photo(image_result)
    if photo is not None:
        bot.send_photo(message.chat.id, photo)
    else:
        bot.send_message(message.chat.id, "Something went wrong 😭")


@bot.message_handler(commands=["gray"])
def gray_image(message):
    bot.send_message(message.chat.id, "Send me an image")
    bot.register_next_step_handler(message, gray_image_next_step)


@bot.message_handler(content_types=["photo"])
def gray_image_next_step(message):
    if message.content_type == "photo":
        image = message_photo_to_image(message)
        image_result = filters.image2gray(image)
        photo = image_to_message_photo(image_result)
        if photo is not None:
            bot.send_photo(message.chat.id, photo)
        else:
            bot.send_message(message.chat.id, "Something went wrong 😭")


@bot.message_handler(commands=["besco"])
def send_pencil_sketch(message):
    bot.send_message(message.chat.id, "Send me an image of your beautiful face")
    bot_state = "besco"


@bot.message_handler(commands=["cartoon"])
def send_pencil_sketch(message):
    bot.send_message(message.chat.id, "Send me an image")
    bot_state = "cartoon"


@bot.message_handler(commands=["inverse"])
def send_pencil_sketch(message):
    bot.send_message(message.chat.id, "Send me an image")
    bot_state = "inverse"


@bot.message_handler(commands=["qrcode"])
def send_qrcode(message):
    msg = bot.send_message(message.chat.id, "Send me a text or url")
    # bot_state = 'qrcode'
    bot.register_next_step_handler(msg, process_qrcode_step)


@bot.message_handler(commands=["style_transfer"])
def send_style_transfer(message):
    photo = open("input/style_transfer.png", "rb")
    bot.send_photo(message.chat.id, photo)
    bot.send_message(message.chat.id, "Send me content image")
    bot_state = "style_transfer_1"


@bot.message_handler(commands=["find_my_face"])
def find_my_face(message):
    bot.send_message(message.chat.id, "Send me your face image")
    user_data[message.chat.id] = {'step': 'awaiting_photo', 'responses': []}
    bot.register_next_step_handler(message, find_my_face_next_step)


def send_images(step, image_results, message):
    numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]
    for index, image in enumerate(image_results):
        # photo = numpy_to_bytes_io(image['nose_image'])
        # types.InputMediaPhoto(open('path/to/image1.jpg', 'rb')),
        # media_group.append(types.InputMediaPhoto(photo))
        photo = image_to_message_photo(image[step])
        if photo is not None:
            # bot.send_media_group(message.chat.id, media_group)
            bot.send_photo(message.chat.id, photo, numbers[index])
        else:
            bot.send_message(message.chat.id, "Something went wrong 😭")
            break


@bot.message_handler(content_types=['photo'])
def find_my_face_next_step(message):
    if message.chat.id in user_data and user_data[message.chat.id]['step'] == 'awaiting_photo':
        if message.content_type == "photo":

            # Save the photo sent by the user (if needed)
            bot.send_message(message.chat.id, "Photo received! Now, I'll send you 4 images. Please choose one by sending a number (1-4).")

            image = message_photo_to_image(message)
            user_data[message.chat.id]['image_results'] = filters.find_my_face(image, face_parts_detector)
            user_data[message.chat.id]['image_step'] = "nose_image"
            send_images(user_data[message.chat.id]['image_step'], user_data[message.chat.id]['image_results'], message)

            # Save message IDs for reference
            user_data[message.chat.id]['step'] = 'awaiting_choice'


@bot.message_handler(func=lambda message: message.chat.id in user_data and user_data[message.chat.id]['step'] == 'awaiting_choice')
def handle_choice(message):
    try:
        choice = int(message.text)
        if 1 <= choice <= 4:
            # Save the user's choice
            user_data[message.chat.id]['responses'].append(choice)
            if len(user_data[message.chat.id]['responses']) < 4:
                # Send 4 new images for the next round
                bot.send_message(message.chat.id, "Choose an image by sending a number (1-4).")
                if user_data[message.chat.id]['image_step'] == "nose_image":
                    user_data[message.chat.id]['image_step'] = "left_eye_image"
                elif user_data[message.chat.id]['image_step'] == "left_eye_image":
                    user_data[message.chat.id]['image_step'] = "right_eye_image"
                elif user_data[message.chat.id]['image_step'] == "right_eye_image":
                    user_data[message.chat.id]['image_step'] = "lips_image"
    
                send_images(user_data[message.chat.id]['image_step'], user_data[message.chat.id]['image_results'], message)
            else:
                # Final result
                bot.send_message(message.chat.id, "Thank you for your choices! Here are your final results:")
                for i, response in enumerate(user_data[message.chat.id]['responses'], start=1):
                    bot.send_message(message.chat.id, f"Round {i}: Image {response}")

                # Clear user data
                del user_data[message.chat.id]
        else:
            bot.send_message(message.chat.id, "Please send a number between 1 and 4.")
    except ValueError:
        bot.send_message(message.chat.id, "Please send a valid number.")


def message_photo_to_image(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_data = np.frombuffer(downloaded_file, dtype=np.uint8)
    image = cv2.imdecode(file_data, 1)
    return image


def image_to_message_photo(image):
    retval, buffer = cv2.imencode(".png", image)
    if retval:
        return buffer
    else:
        return None


def numpy_to_bytes_io(image_np):
    # Encode image as a JPEG file in memory
    _, image_buffer = cv2.imencode(".jpg", image_np)
    # Convert buffer to BytesIO object
    return BytesIO(image_buffer.tobytes())


@bot.message_handler(content_types=["photo"])
def send_photo(message):
    global bot_state, content_image_path

    file_info = bot.get_file(message.photo[-1].file_id)

    if bot_state == "style_transfer_1":
        content_image_path = save_image(file_info, "content_image")
        bot.send_message(message.chat.id, "Send me style image")
        bot_state = "style_transfer_2"
        return

    elif bot_state == "style_transfer_2":
        style_image_path = save_image(file_info, "style_image")
        image_result = styleTransfer(content_image_path, style_image_path)

    elif bot_state == "besco":
        image_path = save_image(file_info)
        image_result = face_eyes_lips(image_path)

    elif bot_state == "cartoon":
        image_path = save_image(file_info)
        image_result = image2cartoon(image_path)

    image_path = os.path.join("output", file_info.file_path)
    cv2.imwrite(image_path, image_result)

    photo = open(image_path, "rb")
    bot.send_photo(message.chat.id, photo)


@bot.message_handler(func=lambda message: True)
def send_message(message):
    try:
        pass

    except:
        bot.send_message(message.chat.id, "Something went wrong 😭")


def process_qrcode_step(message):
    try:
        image_path = "output/photos/qrcode.png"
        img = qrcode.make(message.text)
        img.save(image_path)

        photo = open(image_path, "rb")
        bot.send_photo(message.chat.id, photo)
    except Exception as e:
        bot.send_message(message.chat.id, "Something went wrong 😭")


bot.polling()
