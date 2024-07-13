# C++代码库

## 运行

```bash
# 编译
mkdir build && cd build
cmake ..
make

# 运行测试
./benchmark /path/to/image 重复测试次数（结果取平均值）
./benchmark /home/admin/p1.png 8

# 检测单张图片
./detectImage /path/to/input_image /path/to/output_image
./detectImage /home/admin/p1.png p1_detect.png
```

## 静态库

编译后，共享库文件会存放在 ./lib 文件夹下

- libbase.a   基准库
- libneon.a  neon加速库

两个库函数均提供如下的接口函数

```c++
int *facedetect_cnn(unsigned char *result_buffer, unsigned char *rgb_image_data, int width, int height, int step);

// unsigned char *result_buffer
// memory for storing face detection results, !!its size must be 0x20000 Bytes!!

// input image, it must be BGR (three-channel) image!
```

