import numpy as np
from PIL import Image, ImageDraw

from clib.interface import FaseDetectInterface

# 动态链接库路径
tool = FaseDetectInterface("clib/build/lib")


def process_frame(image: Image.Image, method='neon', threshold=50) -> Image.Image:
    # 将图像转换为 BGR 格式
    image_np = np.array(image, dtype=np.uint8)
    a = image_np[:, :, 0]
    c = image_np[:, :, 2]
    image_np[:, :, 0] = c
    image_np[:, :, 2] = a

    result = tool.detect_faces(image_np, method=method)
    draw = ImageDraw.Draw(image)

    for item in result:
        confidence = item["confidence"]
        if int(confidence) > threshold:
            x1 = item['x']
            y1 = item['y']
            x2 = item['x'] + item['w']
            y2 = item['y'] + item['h']

            draw.rectangle((x1, y1, x2, y2), outline="blue", width=3)
            draw.text((x1 + 5, y1 + 5), f'{item["confidence"]}', fill="red")

    return image
