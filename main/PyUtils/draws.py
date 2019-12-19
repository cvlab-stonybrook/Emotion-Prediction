import os
import psutil
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def drawTextonPIL(image_path, text, font_file=None, font_size=25, rect_x_rate=1.0, rect_y_rate=1.0):
    if font_file is None:
        font_file = '/usr/share/fonts/truetype/Sarai/Sarai.ttf'
    font = ImageFont.truetype(font_file, font_size)

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img, 'RGBA')
    rect_x = img.size[0]
    rect_y = img.size[1]

    # lay on a rectangle covering half the size of the image
    draw.rectangle((0, 0, int(rect_x * rect_x_rate), int(rect_y*rect_y_rate)), fill=(0 ,0 ,128 ,128))

    draw.text((0, 0), text, (255, 255, 0, 0) ,font=font)
    # img.save(os.path.join(save_path, os.path.basename(image_path)))
    return img

def save_image(img, image_path):
    img.save(image_path)


def hide_shown_image():
    for proc in psutil.process_iter():
        if proc.name() == "display":
            proc.kill()

