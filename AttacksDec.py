import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob=0.002, pepper_prob=0.002):
    row, col, ch = image.shape
    noisy = np.copy(image)

    # Salt noise
    num_salt = np.ceil(salt_prob * image.size * 0.5)
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

def calculate_psnr(original, noisy):
    return peak_signal_noise_ratio(original, noisy)

def save_image(image, name):
    cv2.imwrite(name + '.jpg', cv2.cvtColor(image, cv2.COLOR_HSV2BGR))

# Read the original image
image = cv2.imread('watermarked_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Save the original image
save_image(image, 'original')

# Add various types of noise
noisy_image_gaussian = add_gaussian_noise(image)
noisy_image_salt_pepper = add_salt_and_pepper_noise(image)
noisy_image_speckle = add_speckle_noise(image)
noisy_image_poisson = add_poisson_noise(image)

# Save each noisy image with its name
save_image(noisy_image_gaussian, 'gaussian_noise')
save_image(noisy_image_salt_pepper, 'salt_pepper_noise')
save_image(noisy_image_speckle, 'speckle_noise')
save_image(noisy_image_poisson, 'poisson_noise')

# Apply mean filter to noisy images
figure_size = 9
filtered_image_gaussian = cv2.blur(noisy_image_gaussian, (figure_size, figure_size))
filtered_image_salt_pepper = cv2.blur(noisy_image_salt_pepper, (figure_size, figure_size))
filtered_image_speckle = cv2.blur(noisy_image_speckle, (figure_size, figure_size))
filtered_image_poisson = cv2.blur(noisy_image_poisson, (figure_size, figure_size))

# Save each filtered image with its name
save_image(filtered_image_gaussian, 'gaussian_filtered')
save_image(filtered_image_salt_pepper, 'salt_pepper_filtered')
save_image(filtered_image_speckle, 'speckle_filtered')
save_image(filtered_image_poisson, 'poisson_filtered')

# Calculate PSNR for each noisy and filtered image
psnr_gaussian = calculate_psnr(image[..., 2], noisy_image_gaussian[..., 2])
psnr_salt_pepper = calculate_psnr(image[..., 2], noisy_image_salt_pepper[..., 2])
psnr_speckle = calculate_psnr(image[..., 2], noisy_image_speckle[..., 2])
psnr_poisson = calculate_psnr(image[..., 2], noisy_image_poisson[..., 2])

psnr_filtered_gaussian = calculate_psnr(image[..., 2], filtered_image_gaussian[..., 2])
psnr_filtered_salt_pepper = calculate_psnr(image[..., 2], filtered_image_salt_pepper[..., 2])
psnr_filtered_speckle = calculate_psnr(image[..., 2], filtered_image_speckle[..., 2])
psnr_filtered_poisson = calculate_psnr(image[..., 2], filtered_image_poisson[..., 2])

# Print PSNR values on the console
print(f'PSNR for Gaussian Noise: {psnr_gaussian:.2f} dB')
print(f'PSNR for Salt & Pepper Noise: {psnr_salt_pepper:.2f} dB')
print(f'PSNR for Speckle Noise: {psnr_speckle:.2f} dB')
print(f'PSNR for Poisson Noise: {psnr_poisson:.2f} dB')

print(f'PSNR for Gaussian Filtered Image: {psnr_filtered_gaussian:.2f} dB')
print(f'PSNR for Salt & Pepper Filtered Image: {psnr_filtered_salt_pepper:.2f} dB')
print(f'PSNR for Speckle Filtered Image: {psnr_filtered_speckle:.2f} dB')
print(f'PSNR for Poisson Filtered Image: {psnr_filtered_poisson:.2f} dB')

# Display the results
plt.figure(figsize=(15, 12))

plt.subplot(331), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(332), plt.imshow(cv2.cvtColor(noisy_image_gaussian, cv2.COLOR_HSV2RGB)), plt.title('Gaussian Noise')
plt.xticks([]), plt.yticks([])

plt.subplot(333), plt.imshow(cv2.cvtColor(filtered_image_gaussian, cv2.COLOR_HSV2RGB)), plt.title('Gaussian Filtered')
plt.xticks([]), plt.yticks([])

plt.subplot(334), plt.imshow(cv2.cvtColor(noisy_image_salt_pepper, cv2.COLOR_HSV2RGB)), plt.title('Salt & Pepper Noise')
plt.xticks([]), plt.yticks([])

plt.subplot(335), plt.imshow(cv2.cvtColor(filtered_image_salt_pepper, cv2.COLOR_HSV2RGB)), plt.title('Salt & Pepper Filtered')
plt.xticks([]), plt.yticks([])

plt.subplot(336), plt.imshow(cv2.cvtColor(noisy_image_speckle, cv2.COLOR_HSV2RGB)), plt.title('Speckle Noise')
plt.xticks([]), plt.yticks([])

plt.subplot(337), plt.imshow(cv2.cvtColor(filtered_image_speckle, cv2.COLOR_HSV2RGB)), plt.title('Speckle Filtered')
plt.xticks([]), plt.yticks([])

plt.subplot(338), plt.imshow(cv2.cvtColor(noisy_image_poisson, cv2.COLOR_HSV2RGB)), plt.title('Poisson Noise')
plt.xticks([]), plt.yticks([])

plt.subplot(339), plt.imshow(cv2.cvtColor(filtered_image_poisson, cv2.COLOR_HSV2RGB)), plt.title('Poisson Filtered')
plt.xticks([]), plt.yticks([])

plt.show()
