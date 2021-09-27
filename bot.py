import random
import os
import cv2
import telebot
from telebot import types
from filters.image_to_pencil_sketch import *
import config


Token = os.environ["Tokan"]
bot = telebot.TeleBot(Token)

markup = types.ReplyKeyboardMarkup(row_width=4)
itembtn1 = types.KeyboardButton('+')
itembtn2 = types.KeyboardButton('-')
itembtn3 = types.KeyboardButton('*')
itembtn4 = types.KeyboardButton('/')
markup.add(itembtn1, itembtn2, itembtn3, itembtn4)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, config.start_test)


@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, config.help_text)


@bot.message_handler(commands=['image_to_pencil_sketch'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Please send an image")


@bot.message_handler(content_types=['photo'])
def photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image_path = os.path.join('input', file_info.file_path)
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    image_result = image2pencilSketch(image_path)
    image_path = os.path.join('output', file_info.file_path)
    cv2.imwrite(image_path, image_result)

    photo = open(image_path, "rb")
    bot.send_photo(message.chat.id, photo)


@bot.message_handler(func=lambda message: True)
def sajjad(message):
    for letter in message.text:
        if letter != ' ':
            bot.send_message(message.chat.id, letter)

bot.polling()
