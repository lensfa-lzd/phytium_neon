import ctypes
from ctypes import cdll, POINTER, c_int, c_short, c_ubyte

import numpy as np


class FaseDetectInterface(object):
    def __init__(self, lib_path: str):
        # 建议输入绝对路径
        self.lib = cdll.LoadLibrary(f'{lib_path}/libfaceDetect.so')

        """
        // 链接库调用函数
        // unsigned char *result_buffer
        // memory for storing face detection results, !!its size must be 0x800 Bytes!!

        // input image, it must be BGR (three-channel) image!
        short *facedetect_cnn(unsigned char *result_buffer, unsigned char *rgb_image_data, int width, int height, int step);
        """
        self.lib.hello()

        # 定义函数参数和返回值类型
        self.lib.faceDetectBase.argtypes = [
            POINTER(c_ubyte),  # result_buffer
            POINTER(c_ubyte),  # rgb_image_data
            c_int,  # width
            c_int,  # height
            c_int  # step
        ]
        self.lib.faceDetectBase.restype = POINTER(c_short)

        self.lib.faceDetectNeon.argtypes = [
            POINTER(c_ubyte),  # result_buffer
            POINTER(c_ubyte),  # rgb_image_data
            c_int,  # width
            c_int,  # height
            c_int  # step
        ]
        self.lib.faceDetectNeon.restype = POINTER(c_short)

        # 分配结果缓冲区
        self.result_buffer = ctypes.cast(ctypes.create_string_buffer(0x800), POINTER(c_ubyte))

    def detect_faces(self, image_array: np.ndarray, method='base') -> list:
        height, width = image_array.shape[:2]
        step = image_array.strides[0]

        image_data = image_array.ctypes.data_as(POINTER(c_ubyte))

        result_buffer = ctypes.cast(ctypes.create_string_buffer(0x800), POINTER(c_ubyte))
        # 调用 C++ 函数
        if method == 'base':
            result_ptr = self.lib.faceDetectBase(
                result_buffer, image_data, width, height, step
            )
        elif method == 'neon':
            result_ptr = self.lib.faceDetectNeon(
                result_buffer, image_data, width, height, step
            )
        else:
            raise ValueError("Unsupported Method")

        # 获取结果数组长度
        result = []
        array_length = result_ptr[0]
        if array_length > 0:
            # 遍历结果数组
            data_array: list = result_ptr[1: 16 * array_length]
            for i in range(array_length):
                result_array = data_array[i * 16: (1 + i) * 15]
                result.append({
                    'confidence': result_array[0],
                    'x': result_array[1],
                    'y': result_array[2],
                    'w': result_array[3],
                    'h': result_array[4],
                })

        return result
