# C++代码库

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

