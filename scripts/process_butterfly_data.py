import os
import random
import argparse
from pathlib import Path

def process_cbd(base_dir, split_ratio=0.8):
    print(f"Processing CBD dataset at {base_dir}...")
    images_dir = os.path.join(base_dir, 'images')
    masks_dir = os.path.join(base_dir, 'masks')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Error: images or masks directory not found in {base_dir}")
        return

    valid_samples = []
    
    # Walk through images directory to find all images
    # CBD structure: images/SpeciesName/ImageName.jpg
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Get relative path from images_dir, e.g., "SpeciesName/ImageName.jpg"
                rel_path = os.path.relpath(os.path.join(root, file), images_dir)
                
                # We store the path without extension to be flexible, or with extension?
                # U-Bench usually expects the name to be used to construct paths.
                # If we store "SpeciesName/ImageName" (no ext), we need to know ext when loading.
                # Let's store "SpeciesName/ImageName" and handle ext in Dataset class.
                
                name_without_ext = os.path.splitext(rel_path)[0]
                
                # Check if corresponding mask exists
                # Masks seem to be .png in CBD
                # Path: masks/SpeciesName/ImageName.png
                mask_rel_path = name_without_ext + '.png'
                mask_full_path = os.path.join(masks_dir, mask_rel_path)
                
                if os.path.exists(mask_full_path):
                    valid_samples.append(name_without_ext)
                else:
                    # Try other extensions for mask?
                    found = False
                    for ext in ['.jpg', '.jpeg', '.bmp']:
                        if os.path.exists(os.path.join(masks_dir, name_without_ext + ext)):
                            valid_samples.append(name_without_ext)
                            found = True
                            break
                    if not found:
                        # print(f"Warning: Mask not found for {rel_path}")
                        pass

    print(f"Found {len(valid_samples)} valid samples in CBD.")
    
    if len(valid_samples) == 0:
        return

    # Shuffle and split
    # Sort first to ensure deterministic shuffle with seed
    valid_samples.sort()
    random.seed(42)
    random.shuffle(valid_samples)
    
    split_idx = int(len(valid_samples) * split_ratio)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]
    
    train_file = os.path.join(base_dir, 'train.txt')
    val_file = os.path.join(base_dir, 'val.txt')
    
    # Use 'w' to overwrite existing
    with open(train_file, 'w', encoding='utf-8') as f:
        # On Windows, path separator is \, but usually / is preferred for compatibility
        # Python's os.path.join uses \ on Windows. 
        # Let's normalize to / for text files to avoid escape issues in some readers
        f.write('\n'.join([s.replace(os.sep, '/') for s in train_samples]))
        
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join([s.replace(os.sep, '/') for s in val_samples]))
        
    print(f"CBD: Saved {len(train_samples)} training samples to {train_file}")
    print(f"CBD: Saved {len(val_samples)} validation samples to {val_file}")


def process_butterfly200(base_dir):
    print(f"Checking Butterfly200 dataset at {base_dir}...")
    
    images_dir = os.path.join(base_dir, 'images_small')
    if not os.path.exists(images_dir):
        print(f"Butterfly200: images_small not found.")
        return

    # Check for SegmentationClass
    masks_dir = os.path.join(base_dir, 'SegmentationClass')
    if not os.path.exists(masks_dir):
        # Fallback to masks if SegmentationClass not found
        masks_dir = os.path.join(base_dir, 'masks')
    
    if not os.path.exists(masks_dir):
        print("Butterfly200: 'SegmentationClass' or 'masks' directory not found.")
        return

    print(f"Butterfly200: Found mask directory at {masks_dir}")

    valid_samples = []
    
    # Walk through images directory to find all images
    # Structure: images_small/SpeciesName/ImageName.jpg
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Get relative path from images_dir
                rel_path = os.path.relpath(os.path.join(root, file), images_dir)
                
                name_without_ext = os.path.splitext(rel_path)[0]
                
                # Check if corresponding mask exists
                # Masks seem to be .png in SegmentationClass
                # Path: SegmentationClass/SpeciesName/ImageName.png
                
                mask_rel_path = name_without_ext + '.png'
                mask_full_path = os.path.join(masks_dir, mask_rel_path)
                
                if os.path.exists(mask_full_path):
                    valid_samples.append(name_without_ext)
                else:
                    # Try other extensions for mask
                    for ext in ['.jpg', '.jpeg', '.bmp']:
                        if os.path.exists(os.path.join(masks_dir, name_without_ext + ext)):
                            valid_samples.append(name_without_ext)
                            break
    
    print(f"Found {len(valid_samples)} valid samples in Butterfly200.")

    if len(valid_samples) == 0:
        return

    # Shuffle and split
    valid_samples.sort()
    random.seed(42)
    random.shuffle(valid_samples)
    
    # 80/20 split
    split_idx = int(len(valid_samples) * 0.8)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]
    
    train_file = os.path.join(base_dir, 'train.txt')
    val_file = os.path.join(base_dir, 'val.txt')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join([s.replace(os.sep, '/') for s in train_samples]))
        
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join([s.replace(os.sep, '/') for s in val_samples]))
        
    print(f"Butterfly200: Saved {len(train_samples)} training samples to {train_file}")
    print(f"Butterfly200: Saved {len(val_samples)} validation samples to {val_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=r'd:\曲线分割\U-Bench\data\butterfly_dataset', help='Root of butterfly datasets')
    args = parser.parse_args()
    
    cbd_dir = os.path.join(args.data_root, 'cbd')
    bf200_dir = os.path.join(args.data_root, 'butterfly_200')
    
    if os.path.exists(cbd_dir):
        process_cbd(cbd_dir)
    else:
        print(f"CBD directory not found at {cbd_dir}")
        
    if os.path.exists(bf200_dir):
        process_butterfly200(bf200_dir)
    else:
        print(f"Butterfly200 directory not found at {bf200_dir}")
