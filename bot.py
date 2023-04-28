import random
import os
import cv2
import qrcode
import telebot
from telebot import types
from filters import *
from style_transfer import styleTransfer
import config


Token = os.environ["Token"]
bot = telebot.TeleBot(Token)

markup = types.ReplyKeyboardMarkup(row_width=4)
itembtn1 = types.KeyboardButton('+')
itembtn2 = types.KeyboardButton('-')
itembtn3 = types.KeyboardButton('*')
itembtn4 = types.KeyboardButton('/')
markup.add(itembtn1, itembtn2, itembtn3, itembtn4)

bot_state = None
content_image_path = None


@bot.message_handler(commands=['start'])
def send_start(message):
    bot.reply_to(message, config.start_test)


@bot.message_handler(commands=['help'])
def send_help(message):
    bot.reply_to(message, config.help_text)


@bot.message_handler(commands=['pencil_sketch'])
def send_pencil_sketch(message):
    global bot_state
    bot.send_message(message.chat.id, "Send me an image")
    bot_state = 'pencil_sketch'


@bot.message_handler(commands=['gray'])
def send_pencil_sketch(message):
    global bot_state
    bot.send_message(message.chat.id, "Send me an image")
    bot_state = 'gray'


@bot.message_handler(commands=['besco'])
def send_pencil_sketch(message):
    global bot_state
    bot.send_message(message.chat.id, "Send me an image of your beautiful face")
    bot_state = 'besco'


@bot.message_handler(commands=['cartoon'])
def send_pencil_sketch(message):
    global bot_state
    bot.send_message(message.chat.id, "Send me an image")
    bot_state = 'cartoon'


@bot.message_handler(commands=['inverse'])
def send_pencil_sketch(message):
    global bot_state
    bot.send_message(message.chat.id, "Send me an image")
    bot_state = 'inverse'


@bot.message_handler(commands=['qrcode'])
def send_qrcode(message):
    global bot_state
    msg = bot.send_message(message.chat.id, "Send me a text or url")
    # bot_state = 'qrcode'
    bot.register_next_step_handler(msg, process_qrcode_step)


@bot.message_handler(commands=['style_transfer'])
def send_style_transfer(message):
    global bot_state
    photo = open('input/style_transfer.png', "rb")
    bot.send_photo(message.chat.id, photo)
    bot.send_message(message.chat.id, "Send me conent image")
    bot_state = 'style_transfer_1'


def save_image(file_info, image_name=None):
    downloaded_file = bot.download_file(file_info.file_path)

    if image_name:
        image_path = os.path.join('input', 'photos', image_name + '.jpg')
    else:
        image_path = os.path.join('input', file_info.file_path)

    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    return image_path


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

    elif bot_state == 'pencil_sketch':
        image_path = save_image(file_info)
        image_result = image2pencilSketch(image_path)

    elif bot_state == 'besco':
        image_path = save_image(file_info)
        image_result = face_eyes_lips(image_path)

    elif bot_state == 'cartoon':
        image_path = save_image(file_info)
        image_result = image2cartoon(image_path)

    elif bot_state == 'gray':
        image_path = save_image(file_info)
        image_result = image2gray(image_path)

    image_path = os.path.join('output', file_info.file_path)
    cv2.imwrite(image_path, image_result)

    photo = open(image_path, "rb")
    bot.send_photo(message.chat.id, photo)


@bot.message_handler(func=lambda message: True)
def send_message(message):
    global bot_state
    try:
        # if bot_state == 'qrcode':
        #     image_path = 'output/photos/qrcode.png'
        #     img = qrcode.make(message.text)
        #     img.save(image_path)

        #     photo = open(image_path, "rb")
        #     bot.send_photo(message.chat.id, photo)
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
