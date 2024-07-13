import ctypes
import numpy as np
from PIL import Image

# 加载 C++ 库
# 需要绝对路径
lib = ctypes.cdll.LoadLibrary("/home/admin/py_ft/clib/build/lib/libbase.so")

"""
int *facedetect_cnn(unsigned char *result_buffer, unsigned char *rgb_image_data, int width, int height, int step);

// unsigned char *result_buffer
// memory for storing face detection results, !!its size must be 0x20000 Bytes!!

// input image, it must be BGR (three-channel) image!
"""

# 定义函数参数和返回值类型
lib.facedetect_cnn.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # result_buffer
    ctypes.POINTER(ctypes.c_ubyte),  # rgb_image_data
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int  # step
]
lib.facedetect_cnn.restype = ctypes.POINTER(ctypes.c_int)


def detect_faces(image_path):
    """
    使用 C++ 函数检测人脸。

    Args:
        image_path: 图像路径。

    Returns:
        检测结果的 NumPy 数组。
    """

    # 使用 OpenCV 读取图像，注意 OpenCV 默认读取为 BGR 格式
    # 读取图像
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # 将图像转换为 BGR 格式
    image_np = np.array(image, dtype=np.uint8)
    a = image_np[:, :, 0]
    c = image_np[:, :, 2]
    image_np[:, :, 0] = c
    image_np[:, :, 2] = a

    k = image_np.strides
    step = image_np.strides[0]

    image_data = image_np.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    # 分配结果缓冲区
    result_buffer = (ctypes.c_ubyte * 0x20000)()

    # 调用 C++ 函数
    result_ptr = lib.facedetect_cnn(
        result_buffer, image_data, width, height, step
    )

    # 获取结果数组长度
    array_length = result_ptr[0]
    print("得到结果数量: ", array_length)
    if array_length > 0:
        # start_ptr = result_ptr + ctypes.sizeof(ctypes.c_int)
        start_ptr = ctypes.byref(result_ptr, ctypes.sizeof(ctypes.c_int))
        start_ptr = ctypes.cast(start_ptr, ctypes.POINTER(ctypes.c_short))

        # 遍历结果数组
        for i in range(array_length):
            # 计算指向每个 short 元素的指针
            # short *p = ((short *) (pResults + 1)) + 16 * i;
            p = ctypes.byref(start_ptr, ctypes.sizeof(ctypes.c_short) * (16 * i))
            p = ctypes.cast(p, ctypes.POINTER(ctypes.c_short))
            shorts = [p[j] for j in range(16)]
            print(shorts)

            # p += 16 * i
            # 访问 p 指向的值
            # value = p[0]
            #
            # # 使用 value 进行后续操作
            # print(f"Element {i}: {value}")


# 示例用法
detect_faces("/home/admin/p1.png")
