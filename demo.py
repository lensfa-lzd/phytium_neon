import numpy as np
from PIL import Image

from clib.interface import FaseDetectInterface


def detect_faces(image_path):
    # 读取图像
    image = Image.open(image_path).convert("RGB")

    # 将图像转换为 BGR 格式
    image_np = np.array(image, dtype=np.uint8)
    a = image_np[:, :, 0]
    c = image_np[:, :, 2]
    image_np[:, :, 0] = c
    image_np[:, :, 2] = a

    tool = FaseDetectInterface("/home/admin/py_ft/clib/build/lib")
    tool.detect_faces(image_np)


# 示例用法
detect_faces("/home/admin/p1.png")
