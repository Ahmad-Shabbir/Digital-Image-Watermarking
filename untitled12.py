# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:09:55 2023

@author: Shabbir
"""

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
import base64

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

def verify_watermark_signature(message_with_signature, public_key_path, image_path):
    lines = message_with_signature.split('\n')
    if len(lines) >= 2:
        message = lines[0].strip()
        signature = lines[1].replace("Signature: ", "").strip()

        print("Verifying Signature...")
        print(f"Original Message: {message}")
        print(f"Original Signature: {signature}")

        # Load the public key from the specified path
        with open(public_key_path, 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

        # Verify the signature using the public key
        if verify_signature(message, signature, public_key):
            print("Watermark is authentic.")
            # Load the image for further processing if needed
            img = Image.open(image_path)
            # Add further processing code here if needed
        else:
            print("Watermark verification failed. The content may have been tampered with.")
    else:
        print("Invalid watermark format.")

# Specify the path to the public key file
public_key_path = 'public_key.pem'

# Specify the path to the watermarked image
watermarked_image_path = 'watermarked_image.jpg'

# Provide the watermark_with_signature, public_key_path, and image_path for verification
verify_watermark_signature(watermark_with_signature, public_key_path, watermarked_image_path)
