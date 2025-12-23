import cv2
import os

base_dir = './data/butterfly_dataset/cbd'
# Read first few samples from train.txt
with open(os.path.join(base_dir, 'train.txt'), 'r') as f:
    samples = [line.strip() for line in f.readlines()[:5]]

for case in samples:
    img_path = os.path.join(base_dir, 'images', case + '.jpg')
    mask_path = os.path.join(base_dir, 'masks', case + '.png')
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    img_shape = img.shape if img is not None else 'NOT FOUND'
    mask_shape = mask.shape if mask is not None else 'NOT FOUND'
    
    print(f'{case}:')
    print(f'  Image: {img_path} -> {img_shape}')
    print(f'  Mask:  {mask_path} -> {mask_shape}')
    if img is not None and mask is not None:
        if img.shape[:2] != mask.shape[:2]:
            print(f'  *** SIZE MISMATCH! ***')
    print()