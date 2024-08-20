import telebot
import numpy as np
import cv2
from telebot import types
from io import BytesIO

API_TOKEN = '7308353438:AAFqhr7-72x20popP8vfbXCwOrPzmF3S4dU'
bot = telebot.TeleBot(API_TOKEN)

# Dictionary to keep track of user states and responses
user_data = {}

def numpy_to_bytes_io(image_np):
    _, image_buffer = cv2.imencode('.jpg', image_np)
    return BytesIO(image_buffer.tobytes())

def generate_images():
    # Generate 4 images (example images)
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img3 = np.full((100, 100, 3), 128, dtype=np.uint8)
    img4 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return [img1, img2, img3, img4]

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Send me a photo and I'll start the process.")
    user_data[message.chat.id] = {'step': 'awaiting_photo', 'responses': []}

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    if message.chat.id in user_data and user_data[message.chat.id]['step'] == 'awaiting_photo':
        # Save the photo sent by the user (if needed)
        bot.send_message(message.chat.id, "Photo received! Now, I'll send you 4 images. Please choose one by sending a number (1-4).")
        
        # Send 4 images to the user
        images = generate_images()
        media_group = [types.InputMediaPhoto(numpy_to_bytes_io(img)) for img in images]
        msg = bot.send_media_group(message.chat.id, media_group)

        # Save message IDs for reference
        user_data[message.chat.id]['images'] = [msg[0].message_id, msg[1].message_id, msg[2].message_id, msg[3].message_id]
        user_data[message.chat.id]['step'] = 'awaiting_choice'

@bot.message_handler(func=lambda message: message.chat.id in user_data and user_data[message.chat.id]['step'] == 'awaiting_choice')
def handle_choice(message):
    try:
        choice = int(message.text)
        if 1 <= choice <= 4:
            # Save the user's choice
            user_data[message.chat.id]['responses'].append(choice)
            if len(user_data[message.chat.id]['responses']) < 3:
                # Send 4 new images for the next round
                bot.send_message(message.chat.id, "Choose an image by sending a number (1-4).")
                images = generate_images()
                media_group = [types.InputMediaPhoto(numpy_to_bytes_io(img)) for img in images]
                msg = bot.send_media_group(message.chat.id, media_group)

                # Update message IDs for reference
                user_data[message.chat.id]['images'] = [msg[0].message_id, msg[1].message_id, msg[2].message_id, msg[3].message_id]
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

bot.polling()
