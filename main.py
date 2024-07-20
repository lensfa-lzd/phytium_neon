import queue
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from scripts import process_frame


def detect_images(image_path, save_path):
    # 读取图像
    image = Image.open(image_path).convert("RGB")
    image = process_frame(image)
    image.save(save_path)


def detect_video(video_path, output_path, sample=10):
    video = VideoFileClip(video_path)  # 读取视频帧
    frames = []
    for frame in video.iter_frames():
        frames.append(frame)
    frames = frames[::sample]  # 间隔抽样
    processed_frames = []

    process = tqdm(total=len(frames), desc="Processing frames")
    start_time = time.time()
    result_queue = queue.Queue()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for frame in frames:
            executor.submit(lambda d: result_queue.put(process_frame(Image.fromarray(d))), frame)

        # 获取结果
        for _ in range(len(frames)):
            processed_frame = result_queue.get()
            processed_frames.append(np.array(processed_frame, dtype=np.uint8))
            process.update(1)

    end_time = time.time()
    process_time = end_time - start_time

    print('\n')
    print(f"处理时间: {process_time:.2f}s, FPS {len(frames) / process_time:.2f}")
    print()

    # 保存处理后的视频帧
    clip = ImageSequenceClip(processed_frames, fps=int(25/sample))
    clip.write_videofile(output_path)


if __name__ == '__main__':
    # 示例用法
    # detect_images("/home/kylin/p1.png", "/home/kylin/p1_detect_py.png")
    detect_video("/home/kylin/videoplayback.mp4", "/home/kylin/videoplayback_detect_py.mp4")
