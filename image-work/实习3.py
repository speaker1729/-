from skimage import morphology, io, color, exposure, restoration
from skimage.morphology import disk
from skimage.filters import unsharp_mask, median
from PIL import Image
import matplotlib.pyplot as plt

image_path = '实习3.tif'
image = Image.open(image_path)
image_array = io.imread(image_path)
if len(image_array.shape) > 2:
    image_gray = color.rgb2gray(image_array)
else:
    image_gray = image_array
image_normalized = exposure.rescale_intensity(image_gray, out_range=(0, 1))

denoised_image = median(image_normalized, morphology.disk(1))

selem = disk(2)

opened_image = morphology.opening(denoised_image, selem)
closed_image = morphology.closing(opened_image, selem)

sharpened_full_image = unsharp_mask(closed_image, radius=1, amount=1.5)

dilated = morphology.dilation(sharpened_full_image, selem)
eroded = morphology.erosion(sharpened_full_image, selem)
morphological_gradient = dilated - eroded
top_hat = morphology.white_tophat(sharpened_full_image, selem)
enhanced_edges = morphological_gradient + top_hat

mix_ratio = 0.7
mixed_image = mix_ratio * enhanced_edges + (1 - mix_ratio) * sharpened_full_image

fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
ax = ax.ravel()

ax[0].imshow(image_normalized, cmap='gray')
ax[0].axis('off')
ax[0].set_title('Original Image')

ax[1].imshow(mixed_image, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Processed Result')

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()
