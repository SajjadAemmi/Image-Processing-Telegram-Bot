from math import e
import random
import os
import cv2
import numpy as np
import qrcode
from scipy.datasets import face
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
itembtn1 = types.KeyboardButton('+')
itembtn2 = types.KeyboardButton('-')
itembtn3 = types.KeyboardButton('*')
itembtn4 = types.KeyboardButton('/')
markup.add(itembtn1, itembtn2, itembtn3, itembtn4)

bot_state = None
content_image_path = None


face_parts_detector = FacePartsDetector()


@bot.message_handler(commands=['start'])
def send_start(message):
    bot.send_message(message.chat.id, config.start_test)


@bot.message_handler(commands=['help'])
def send_help(message):
    bot.send_message(message.chat.id, config.help_text)


@bot.message_handler(commands=['pencil_sketch'])
def pencil_sketch_image(message):
    bot.send_message(message.chat.id, "Send me an image")


@bot.message_handler(content_types=['photo'])
def pencil_sketch_image_next_step(message):
    image = message_photo_to_image(message)
    image_result = filters.image2pencilSketch(image)
    photo = image_to_message_photo(image_result)
    if photo is not None:
        bot.send_photo(message.chat.id, photo)
    else:
        bot.send_message(message.chat.id, "Something went wrong 😭")


@bot.message_handler(commands=['gray'])
def gray_image(message):
    bot.send_message(message.chat.id, "Send me an image")
    bot.register_next_step_handler(message, gray_image_next_step)


@bot.message_handler(content_types=['photo'])
def gray_image_next_step(message):
    if message.content_type == 'photo':
        image = message_photo_to_image(message)
        image_result = filters.image2gray(image)
        photo = image_to_message_photo(image_result)
        if photo is not None:
            bot.send_photo(message.chat.id, photo)
        else:
            bot.send_message(message.chat.id, "Something went wrong 😭")


@bot.message_handler(commands=['besco'])
def send_pencil_sketch(message):
    bot.send_message(message.chat.id, "Send me an image of your beautiful face")
    bot_state = 'besco'


@bot.message_handler(commands=['cartoon'])
def send_pencil_sketch(message):
    bot.send_message(message.chat.id, "Send me an image")
    bot_state = 'cartoon'


@bot.message_handler(commands=['inverse'])
def send_pencil_sketch(message):
    bot.send_message(message.chat.id, "Send me an image")
    bot_state = 'inverse'


@bot.message_handler(commands=['qrcode'])
def send_qrcode(message):
    msg = bot.send_message(message.chat.id, "Send me a text or url")
    # bot_state = 'qrcode'
    bot.register_next_step_handler(msg, process_qrcode_step)


@bot.message_handler(commands=['qrcode'])
def face_puzzle(message):
    msg = bot.send_message(message.chat.id, "Send me a text or url")
    # bot_state = 'qrcode'
    bot.register_next_step_handler(msg, process_qrcode_step)


@bot.message_handler(commands=['style_transfer'])
def send_style_transfer(message):
    photo = open('input/style_transfer.png', "rb")
    bot.send_photo(message.chat.id, photo)
    bot.send_message(message.chat.id, "Send me content image")
    bot_state = 'style_transfer_1'


@bot.message_handler(commands=['who_knows_me_best'])
def who_knows_me_best(message):
    bot.send_message(message.chat.id, "Send me your face image")
    bot.register_next_step_handler(message, who_knows_me_best_next_step)


@bot.message_handler(content_types=['photo'])
def who_knows_me_best_next_step(message):
    if message.content_type == 'photo':
        image = message_photo_to_image(message)
        image_result = filters.who_knows_me_best(image, face_parts_detector)
        photo = image_to_message_photo(image_result)
        if photo is not None:
            bot.send_photo(message.chat.id, photo)
        else:
            bot.send_message(message.chat.id, "Something went wrong 😭")


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


@bot.message_handler(content_types=['photo'])
def send_photo(message):
    global bot_state, content_image_path

    file_info = bot.get_file(message.photo[-1].file_id)

    if bot_state == 'style_transfer_1':
        content_image_path = save_image(file_info, 'content_image')
        bot.send_message(message.chat.id, "Send me style image")
        bot_state = 'style_transfer_2'
        return

    elif bot_state == 'style_transfer_2':
        style_image_path = save_image(file_info, 'style_image')
        image_result = styleTransfer(content_image_path, style_image_path)

    elif bot_state == 'besco':
        image_path = save_image(file_info)
        image_result = face_eyes_lips(image_path)

    elif bot_state == 'cartoon':
        image_path = save_image(file_info)
        image_result = image2cartoon(image_path)

    image_path = os.path.join('output', file_info.file_path)
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
        image_path = 'output/photos/qrcode.png'
        img = qrcode.make(message.text)
        img.save(image_path)

        photo = open(image_path, "rb")
        bot.send_photo(message.chat.id, photo)
    except Exception as e:
        bot.send_message(message.chat.id, "Something went wrong 😭")


bot.polling()
