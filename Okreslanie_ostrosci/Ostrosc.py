import cv2
import numpy as np




def measure_sharpness_L(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian


def measure_blur(image):
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    blur_metric = np.sum(magnitude_spectrum) / np.size(magnitude_spectrum)
    return blur_metric







#image = cv2.imread('kot.jpg', cv2.IMREAD_GRAYSCALE)
#image_2 = cv2.imread('kot_rozmyty.jpg', cv2.IMREAD_GRAYSCALE)

image = cv2.imread('twarz_ostre.png', cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread('twarz_rozmyte.png', cv2.IMREAD_GRAYSCALE)


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