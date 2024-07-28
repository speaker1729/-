import cv2
from skimage import morphology
from skimage.util import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import bm3d

# 读取图像
original_img_path = 'first.jpg'
img = cv2.imread(original_img_path)
original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 将图像数据类型转换为 float32，以便使用 BM3D
img_float = original_img_rgb.astype(np.float32) / 255

# 使用 BM3D 去噪算法
bm3d_denoised_img = bm3d.bm3d(img_float, sigma_psd=30/255)

# 将去噪后的图像转换回 uint8 类型
bm3d_denoised_img_uint8 = img_as_ubyte(bm3d_denoised_img)

# 对比度调整 - 使用CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab = cv2.cvtColor(bm3d_denoised_img_uint8, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
l_clahe = clahe.apply(l)
enhanced_lab = cv2.merge((l_clahe, a, b))
contrast_enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

# 锐化 - 使用高斯模糊和加权方法进行Unsharp Masking
gaussian_blurred = cv2.GaussianBlur(contrast_enhanced_img, (3, 3), 0)
unsharp_image = cv2.addWeighted(contrast_enhanced_img, 1.5, gaussian_blurred, -0.5, 0)

# 色彩增强 - 提高饱和度
hsv = cv2.cvtColor(unsharp_image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv)
s = cv2.add(s, 15)
enhanced_hsv = cv2.merge([h, s, v])
color_enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

# 形态学处理 - 使用形态学梯度突出边缘
gray = cv2.cvtColor(color_enhanced_img, cv2.COLOR_RGB2GRAY)
selem = morphology.disk(3)
dilated = morphology.dilation(gray, selem)
eroded = morphology.erosion(gray, selem)
gradient = dilated - eroded
gradient = np.stack((gradient,)*3, axis=-1)

# 形态学处理结果与色彩增强图像融合
morphology_enhanced_img = cv2.addWeighted(color_enhanced_img, 0.8, gradient, 0.2, 0)

# 超分辨率重建 - 双三次插值上采样
upscaled_img = cv2.resize(morphology_enhanced_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# 显示原始图像和优化后的图像
plt.figure(figsize=(10, 20))
plt.subplot(1, 2, 1)
plt.imshow(original_img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(upscaled_img)
plt.title('Optimized Image')
plt.axis('off')

plt.show()

# 将处理后的图像转换为PIL图像格式
final_optimized_image_pil = Image.fromarray(upscaled_img)

# 保存优化后的图像
final_optimized_image_path = 'shixi1_final_optimized.jpg'
final_optimized_image_pil.save(final_optimized_image_path)
