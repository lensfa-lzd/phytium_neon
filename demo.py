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

    # 检查数组在内存中是否连续存储
    print(image_np.flags['C_CONTIGUOUS'])

    # 动态链接库路径
    # 项目的data文件夹存放了一个在arm中编译好的动态库,可以直接使用
    tool = FaseDetectInterface("data")

    # result = tool.detect_faces(image_np, method='base')
    result = tool.detect_faces(image_np, method='neon')

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


if __name__ == '__main__':
    # 示例用法
    main("/home/admin/p1.png", "/home/admin/p1_detect_py.png")
