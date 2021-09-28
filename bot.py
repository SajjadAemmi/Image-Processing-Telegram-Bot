import random
import os
import cv2
import telebot
from telebot import types
from filters import *
import config
import qrcode


Token = os.environ["Token"]
bot = telebot.TeleBot(Token)

markup = types.ReplyKeyboardMarkup(row_width=4)
itembtn1 = types.KeyboardButton('+')
itembtn2 = types.KeyboardButton('-')
itembtn3 = types.KeyboardButton('*')
itembtn4 = types.KeyboardButton('/')
markup.add(itembtn1, itembtn2, itembtn3, itembtn4)

bot_state = None

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


@bot.message_handler(commands=['inverse'])
def send_pencil_sketch(message):
    global bot_state
    bot.send_message(message.chat.id, "Send me an image")
    bot_state = 'inverse'


@bot.message_handler(commands=['qrcode'])
def send_qrcode(message):
    global bot_state
    bot.send_message(message.chat.id, "Send me your text or url")
    bot_state = 'qrcode'


@bot.message_handler(content_types=['photo'])
def send_photo(message):
    global bot_state

    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image_path = os.path.join('input', file_info.file_path)
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)
        
    if bot_state == 'pencil_sketch':
        image_result = image2pencilSketch(image_path)
    
    elif bot_state == 'gray':
        image_result = image2gray(image_path)


    image_path = os.path.join('output', file_info.file_path)
    cv2.imwrite(image_path, image_result)

    photo = open(image_path, "rb")
    bot.send_photo(message.chat.id, photo)


@bot.message_handler(func=lambda message: True)
def send_message(message):
    global bot_state
    try:
        if bot_state == 'qrcode':
            image_path = 'output/photo/qrcode.png'
            img = qrcode.make(message.text)
            img.save(image_path)

            photo = open(image_path, "rb")
            bot.send_photo(message.chat.id, photo)
    
    except:
        bot.send_message(message.chat.id, "Something went wrong")


bot.polling()
