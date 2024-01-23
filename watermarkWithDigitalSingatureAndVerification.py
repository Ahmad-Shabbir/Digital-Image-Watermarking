# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:10:18 2023

@author: Shabbir
"""

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import PIL
from PIL import Image
import cv2
import numpy as np
from math import log10, sqrt
import matplotlib.pyplot as plt
import sys

def PSNR(original, compressed):
    original = original.astype(np.float64) / 255.
    compressed = compressed.astype(np.float64) / 255.
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def select_roi(image):
    print("Specify the region of interest for watermark embedding.")
    img_width, img_height = image.size
    x_start = int(input("Enter the starting X coordinate of the ROI: "))
    y_start = int(input("Enter the starting Y coordinate of the ROI: "))
    width = int(input("Enter the width of the ROI: "))
    height = int(input("Enter the height of the ROI: "))
    if x_start < 0 or y_start < 0 or x_start + width > img_width or y_start + height > img_height:
        print("Error: Invalid ROI coordinates. Make sure the ROI is within the image boundaries.")
        sys.exit(1)
    return x_start, y_start, width, height

def contrastSensitivity(image, roi):
    print("Calculating contrast sensitivity.......")
    x_start, y_start, width, height = roi
    output_image = image.crop((x_start, y_start, x_start + width, y_start + height))
    for x in range(output_image.width):
        for y in range(output_image.height):
            grayImgArr[y][x] = output_image.getpixel((x, y))[0]
            weber[y][x] = float(abs(grayImgArr[y][x] - 128) / 128)

def entropy(signal):
    lensig = signal.size
    symset = list(set(signal))
    numsym = len(symset)
    propab = [np.size(signal[signal == i]) / (1.0 * lensig) for i in symset]
    ent = np.sum([p * np.log2(1.0 / p) for p in propab])
    return ent

def entropy_H(roi):
    print("Calculating entropy.......")
    region = grayImgArr[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]].flatten()
    H[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = entropy(region)

def visualFactor():
    print("Calculating visual factor.......")
    for x in range(img.width):
        for y in range(img.height):
            J[y][x] = weber[y][x] * H[y][x]

def sign_message(message, private_key):
    signature = private_key.sign(
        message.encode('utf-8'),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

def verify_signature(message, signature, public_key):
    try:
        print(f"Verifying Signature with Message: {message}")
        print(f"Provided Signature: {signature}")
        public_key.verify(
            base64.b64decode(signature.encode('utf-8')),
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception as e:
        print(f"Signature verification failed: {e}")
        return False

def verify_watermark_signature(message_with_signature, public_key):
    lines = message_with_signature.split('\n')
    if len(lines) >= 2:
        message = lines[0].strip()
        signature = lines[1].replace("Signature: ", "").strip()

        print("Verifying Signature...")
        print(f"Original Message: {message}")
        print(f"Original Signature: {signature}")

        # Verify the signature using the public key
        if verify_signature(message, signature, public_key):
            print("Watermark is authentic.")
        else:
            print("Watermark verification failed. The content may have been tampered with.")
    else:
        print("Invalid watermark format.")

def embeddingWatermarkWithSignature(a, b, c, d, roi, feather_size, private_key):
    print("Embedding watermark with digital signature in the specified region...")

    # Generate a unique message
    message = f"Watermarking Message: {roi}"

    # Sign the message with the private key
    signature = sign_message(message, private_key)

    # Embed the digital signature directly into the message
    message_with_signature = f"{message}\nSignature: {signature}"

    # Resize the watermark to fit the ROI
    resized_logo = cv2.resize(logoArr, (roi[2], roi[3]))

    # Create a feathered mask
    feather_mask = np.zeros_like(resized_logo, dtype=np.float32)
    center_x, center_y = roi[2] // 2, roi[3] // 2
    for x in range(roi[2]):
        for y in range(roi[3]):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            feather_mask[y, x] = 1.0 if distance < feather_size / 2 else 1.0 - (distance - feather_size / 2) / (feather_size / 2)

    # Normalize the feather mask
    feather_mask /= feather_mask.max()

    # Initialize the final image array
    final_image = np.array(img)

    # Iterate over the pixels within the ROI
    for x in range(roi[0], roi[0] + roi[2]):
        for y in range(roi[1], roi[1] + roi[3]):
            # Calculate the alpha and beta values using the contrast sensitivity and visual factor
            alpha[y][x] = (b - a) * (J[y][x] - np.amin(J)) / (np.amax(J) - np.amin(J)) + a
            beta[y][x] = (d - c) * (J[y][x] - np.amin(J)) / (np.amax(J) - np.amin(J)) + c

            # Blend the watermark with the original image pixel using the alpha and beta values
            final_image[y, x] = alpha[y][x] * imgArr[y, x] + beta[y][x] * (
                alpha[y][x] * resized_logo[y - roi[1], x - roi[0]] +
                (1 - alpha[y][x]) * resized_logo[y - roi[1], x - roi[0]] * feather_mask[y - roi[1], x - roi[0]]
            )

    # Convert the final image to uint8
    final_image = final_image.astype(np.uint8)

    # Save the watermarked image
    watermarked_image = Image.fromarray(final_image)
    watermarked_image.save("watermarked_image.jpg")
    print("Watermark embedded with digital signature successfully.")

    # Return the watermarked message with the signature for later verification
    return message_with_signature



# Load logo and image
img1 = cv2.imread('waterm.png')
ht, wd, cc = img1.shape
img2 = cv2.imread("lena.jpg")
h_img, w_img, _ = img2.shape
ww = w_img
hh = h_img
color = (255, 255, 255)
result = np.full((hh, ww, cc), color, dtype=np.uint8)
xx = (ww - wd) // 2
yy = (hh - ht) // 2
result[yy:yy + ht, xx:xx + wd] = img1
cv2.imwrite("mylogo_padded.jpg", result)

img = Image.open("lena.jpg")
imgArr = np.array(img)
logo = Image.open("mylogo_padded.jpg")
logoArr = np.array(logo)
gray = img.convert("L")
grayImgArr = np.array(gray)
weber = np.array(gray).astype(float)
H = np.array(gray).astype(float)
J = np.array(gray).astype(float)
alpha = np.array(gray).astype(float)
beta = np.array(gray).astype(float)

# Allow the user to specify ROIs for watermark embedding
roi = select_roi(img)

# Main execution
contrastSensitivity(img, roi)
entropy_H(roi)
visualFactor()

# Generate a private/public key pair for digital signature
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)
public_key = private_key.public_key()

# Embed watermark with digital signature in the specified region
watermark_with_signature = embeddingWatermarkWithSignature(0.7, 0.8, 0.25, 0.116, roi, feather_size=31, private_key=private_key)

# Verify the watermarked image with the signature using the public key
verify_watermark_signature(watermark_with_signature, public_key)

