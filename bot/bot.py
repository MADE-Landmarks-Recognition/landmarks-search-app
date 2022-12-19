import os
import requests
from PIL import Image
from aiogram import Bot, Dispatcher, executor, filters, types


API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

# config
IMG_DIR = "../data/filtered/"                    # common data for all services
UPL_DIR = "./data/uploaded/"                     # upload dir just for web part
TOP_K = 5                                        # api service provide top 5 now
API_ENDPOINT = os.getenv("API_ENDPOINT_LOCAL")   # local api for testing


if not os.path.exists(UPL_DIR):
    os.makedirs(UPL_DIR)


bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


def get_image(path):
    image = Image.open(path).convert("RGB")
    return image


def get_top_similar(image, k):
    """
        send request to landmarks-api service
            {
                "top_k": top k images (now provided top 5)
                "size": image size (need for recovering image from bytes)
                "image": image in bytes
            }
        get top similar json
            {
                "ids": [],
                "names": [],
                "paths": []
            }
    """
    data = {
        'top_k': k,
        'size': image.size,
        'image': image.tobytes().decode("latin-1")
    }
    response = requests.post(url=API_ENDPOINT, json=data, timeout=20)
    top_similar = response.json()
    return top_similar


@dp.message_handler(filters.CommandStart())
async def send_welcome(message: types.Message):
    greeting = f"Привет, {message.from_user.first_name}!\n"
    msg_info_1 = f"Загрузи фотку достопримечательности, и я попробую ее определить или найти похожие на нее.\n"
    msg_info_2 = "Ответ в формате: 5 картинок + подписи к ним, 'id': 'название'"
    await message.reply(greeting + msg_info_1 + msg_info_2)


@dp.message_handler(content_types=['photo'])
async def get_landmarks(message: types.Message):

    PhotoSize = message.photo[-1]
    file_info = await bot.get_file(PhotoSize.file_id)

    if file_info.file_path.lower().endswith(('.png', '.jpg', '.jpeg')):

        await message.reply("Секундочку...")

        await PhotoSize.download(destination_dir=UPL_DIR)

        save_path = UPL_DIR + file_info.file_path
        image = get_image(save_path)
        top_similar = get_top_similar(image, k=TOP_K)

        # Good bots should send chat actions...
        await types.ChatActions.upload_photo()

        media = types.MediaGroup()

        for idx in range(TOP_K):
            media.attach_photo(
                types.InputFile(top_similar["paths"][idx]),
                str(top_similar["ids"][idx]) + ": " + top_similar["names"][idx].strip().replace("_", " ")
            )

        # Done! Send media group
        await message.answer_media_group(media=media)
    
    else:
        await message.answer('Нужно загрузить фото в одном из форматов: "jpg", "png", "jpeg"')



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
