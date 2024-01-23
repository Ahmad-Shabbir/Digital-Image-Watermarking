import numpy as np
import cv2
from matplotlib import pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=50):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob=0.000005,prob=0.0002):
    row, col, ch = image.shape
    noisy = np.copy(image)

    # Salt noise
    num_salt = np.ceil(salt_prob * image.size * 0.02)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords] = 0

    return noisy.astype(np.uint8)

def add_speckle_noise(image, scale=0.1):
    row, col, ch = image.shape
    speckle = scale * np.random.randn(row, col, ch)
    noisy = image + image * speckle
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy.astype(np.uint8)

def adjust_brightness(image, factor=1.5):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def adjust_contrast(image, factor=1.5):
    return np.clip(128 + factor * (image - 128), 0, 255).astype(np.uint8)

def add_random_noise(image, strength=30):
    row, col, ch = image.shape
    random_noise = np.random.normal(0, strength, (row, col, ch))
    noisy = image + random_noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def calculate_psnr(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def save_image(image, name):
    cv2.imwrite(name + '.jpg', cv2.cvtColor(image, cv2.COLOR_HSV2BGR))

# Read the original image
original_image = cv2.imread('watermarked_image.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

# Add various types of attacks
attacked_image_brightness = adjust_brightness(original_image)
attacked_image_contrast = adjust_contrast(original_image)
attacked_image_random_noise = add_random_noise(original_image)

# Calculate PSNR for each attacked image
psnr_brightness = calculate_psnr(original_image[..., 2], attacked_image_brightness[..., 2])
psnr_contrast = calculate_psnr(original_image[..., 2], attacked_image_contrast[..., 2])
psnr_random_noise = calculate_psnr(original_image[..., 2], attacked_image_random_noise[..., 2])

# Save each attacked image with its name
save_image(original_image, 'original')
save_image(attacked_image_brightness, 'brightness_attack')
save_image(attacked_image_contrast, 'contrast_attack')
save_image(attacked_image_random_noise, 'random_noise_attack')

# Display the results
plt.figure(figsize=(15, 8))

plt.subplot(241), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_HSV2RGB)), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(242), plt.imshow(cv2.cvtColor(attacked_image_brightness, cv2.COLOR_HSV2RGB)), plt.title(f'Brightness Attack\nPSNR: {psnr_brightness:.2f}')
plt.xticks([]), plt.yticks([])

plt.subplot(243), plt.imshow(cv2.cvtColor(attacked_image_contrast, cv2.COLOR_HSV2RGB)), plt.title(f'Contrast Attack\nPSNR: {psnr_contrast:.2f}')
plt.xticks([]), plt.yticks([])

plt.subplot(244), plt.imshow(cv2.cvtColor(attacked_image_random_noise, cv2.COLOR_HSV2RGB)), plt.title(f'Random Noise Attack\nPSNR: {psnr_random_noise:.2f}')
plt.xticks([]), plt.yticks([])

# Print PSNR values on the console
print(f'PSNR for Brightness Attack: {psnr_brightness:.2f} dB')
print(f'PSNR for Contrast Attack: {psnr_contrast:.2f} dB')
print(f'PSNR for Random Noise Attack: {psnr_random_noise:.2f} dB')

plt.show()
