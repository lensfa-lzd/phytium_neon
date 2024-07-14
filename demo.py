import numpy as np
from PIL import Image, ImageDraw

from clib.interface import FaseDetectInterface


def main(image_path, save_path):
    # 读取图像
    image = Image.open(image_path).convert("RGB")

    # 将图像转换为 BGR 格式
    image_np = np.array(image, dtype=np.uint8)
    a = image_np[:, :, 0]
    c = image_np[:, :, 2]
    image_np[:, :, 0] = c
    image_np[:, :, 2] = a

    # 动态链接库路径
    tool = FaseDetectInterface("/home/kylin/py_ft/clib/build/lib")

    # result = tool.detect_faces(image_np, method='base')  # 普通版
    result = tool.detect_faces(image_np, method='neon')  # neon 加速

    # 创建可绘制对象
    draw = ImageDraw.Draw(image)

    for item in result:
        x1 = item['x']
        y1 = item['y']
        x2 = item['x'] + item['w']
        y2 = item['y'] + item['h']

        # 绘制方框
        draw.rectangle((x1, y1, x2, y2), outline="blue", width=3)

        # 绘制置信度
        draw.text((x1 + 5, y1 + 5), f'{item["confidence"]}', fill="red")

    image.save(save_path)


# 示例用法
main("/home/kylin/p1.png", "/home/kylin/p1_detect_py.png")
