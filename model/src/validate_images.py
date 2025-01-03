import os
from PIL import Image

def validate_images(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, filename)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except Exception as e:
                    print(f"Corrupted image found: {filepath}")
                    print(f"Error: {str(e)}")

if __name__ == "__main__":
    validate_images("Dataset")
