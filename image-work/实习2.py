from PIL import Image
import numpy as np
import os  # 用于获取文件大小


def downsample_image(image_data, factor=2):
    """
    降低图像分辨率。
    """
    new_height = image_data.shape[0] // factor
    new_width = image_data.shape[1] // factor
    downsampled_image_data = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            block = image_data[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor]
            downsampled_image_data[y, x] = block.mean(axis=(0, 1))

    return downsampled_image_data


def rgb_to_grayscale(image_data):
    """
    RGB到灰度的颜色空间转换。
    """
    return np.dot(image_data[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def quantize(image_data, levels=16):
    """
    将灰度值量化到指定的等级。
    """
    quantized = np.floor(image_data / (256 / levels)) * (256 / levels)
    return quantized.astype(np.uint8)


# 加载原始图像
image_path = '实习2.bmp'
original_image = Image.open(image_path)
original_image_data = np.array(original_image)

# 应用降采样
downsampled_image_data = downsample_image(original_image_data, factor=2)

# 转换到灰度
grayscale_data = rgb_to_grayscale(downsampled_image_data)

# 应用量化
quantized_data = quantize(grayscale_data, 16)

# 将处理后的数据转换回PIL图像以便保存
processed_image = Image.fromarray(quantized_data)

# 保存处理后的图像
compressed_image_path = 'compressed_image.bmp'
processed_image.save(compressed_image_path, 'BMP')

# 获取原始和压缩后的图像文件大小
original_size = os.path.getsize(image_path)
compressed_size = os.path.getsize(compressed_image_path)

# 计算压缩比
compression_ratio = original_size / compressed_size

print(f"Original Size: {original_size} bytes")
print(f"Compressed Size: {compressed_size} bytes")
print(f"Compression Ratio: {compression_ratio:.2f}")
