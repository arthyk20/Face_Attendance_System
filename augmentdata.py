import os
import imageio
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from tqdm import tqdm

# Define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% chance to flip horizontally
    iaa.Affine(
        rotate=(-10, 10),            # Rotate ±10 degrees
        scale=(0.9, 1.1),            # Zoom
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
    ),
    iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)),  # Light noise
    iaa.GaussianBlur(sigma=(0, 1.0)),                # Slight blur
    iaa.Multiply((0.8, 1.2))                         # Brightness
])

# Set input and output directories
input_root = 'D:/FaceAttendence/Dataset'
output_root = 'D:/FaceAttendence/augmented_data'
num_augments = 10  # Number of augmented images per original

# Create output folder
if not os.path.exists(output_root):
    os.makedirs(output_root)

# Loop through each person's folder
for person in tqdm(os.listdir(input_root), desc="Augmenting persons"):
    person_path = os.path.join(input_root, person)
    if not os.path.isdir(person_path):
        continue

    output_person_path = os.path.join(output_root, person)
    os.makedirs(output_person_path, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        try:
            image = imageio.imread(img_path)

            for i in range(num_augments):
                aug_image = seq(image=image)
                aug_img_pil = Image.fromarray(aug_image)

                aug_img_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
                aug_img_pil.save(os.path.join(output_person_path, aug_img_name))

        except Exception as e:
            print(f"Failed on {img_path}: {e}")