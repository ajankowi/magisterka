import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



def measure_sharpness_L(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian


def measure_blur(image):
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    blur_metric = np.sum(magnitude_spectrum) / np.size(magnitude_spectrum)
    return blur_metric




image_folder = '/home/adam_j4/magisterka/Okreslanie_ostrosci/Ostre'
images = []

for filename in os.listdir(image_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)

           

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, image in enumerate(images):
    sharpness = measure_sharpness_L(image)
    blur_score = measure_blur(image)
    row = i // 3
    col = i % 3
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(f"Sharpness L: {sharpness}\nSharpness FFT: {blur_score}")
    axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()




for image in images:
    sharpness = measure_sharpness_L(image)
    print(f"Sharpness L: {sharpness}")
    blur_score = measure_blur(image)
    print(f"Sharpness FFT: {blur_score}")


'''
# Porównanie pierwszego sposobu - filtr laplace
sharpness = measure_sharpness_L(image)
print(f"Sharpness L: {sharpness}")

sharpness = measure_sharpness_L(image_2)
print(f"Sharpness_2 L: {sharpness}")



# Porównanie 2 sposobu - FFT
if image is None:
    print('Image not loaded')
else:
    blur_score = measure_blur(image)
    print(f"Sharpness FFT: {blur_score}")


if image_2 is None:
    print('Image not loaded')
else:
    blur_score_2 = measure_blur(image_2)
    print(f"Sharpness FFT_2: {blur_score_2}")
    '''