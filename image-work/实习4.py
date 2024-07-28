from skimage import io, color, exposure, filters, morphology, measure, util
import matplotlib.pyplot as plt
import numpy as np

# 图像文件路径
image_path = '实习4.jpg'

# 加载图像
image = io.imread(image_path)

# 转换为灰度图像
gray_image = color.rgb2gray(image)

# 应用直方图均衡化增强对比度
equalized_image = exposure.equalize_adapthist(gray_image, clip_limit=0.02)

# 使用Sobel算子进行边缘检测
edges_sobel = filters.sobel(equalized_image)

# 使用Multi-Otsu算子进行阈值分割
threshold_value = filters.threshold_multiotsu(edges_sobel, classes=3)
binary_image = edges_sobel > threshold_value[1]

# 形态学处理
selem = morphology.disk(3)
closed_image = morphology.binary_closing(binary_image, selem)

# 去除小的连通区域
labeled_image = measure.label(closed_image)
props = measure.regionprops(labeled_image)
for prop in props:
    if prop.area < 100:
        labeled_image[labeled_image == prop.label] = 0
cleaned_image = labeled_image > 0

# 骨架化提取中心线
skeleton = morphology.skeletonize(cleaned_image)

# 保存处理后的图像
processed_image_path = 'processed_retinal_image.png'
io.imsave(processed_image_path, util.img_as_ubyte(skeleton))

# 展示图像
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # 更新为展示两个图像
axes_flat = axes.flatten()

titles = ['Original Image', 'Skeleton Image']
images = [image, skeleton]

for idx, ax in enumerate(axes_flat):
    cmap = 'gray' if idx > 0 else None
    ax.imshow(images[idx], cmap=cmap)
    ax.axis('off')
    ax.set_title(titles[idx])

plt.tight_layout()
plt.show()
