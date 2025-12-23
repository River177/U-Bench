import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class CBDDataset(Dataset):
    def __init__(
        self,
        base_dir=None,
        mode="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.mode = mode
        self.transform = transform
        
        if self.mode == "train":
            file_path = os.path.join(self._base_dir, train_file_dir)
        elif self.mode == "val":
            file_path = os.path.join(self._base_dir, val_file_dir)
        elif self.mode == "test":
             # U-Bench often uses val or test split. If test.txt exists, use it, else val.txt
             test_path = os.path.join(self._base_dir, "test.txt")
             if os.path.exists(test_path):
                 file_path = test_path
             else:
                 file_path = os.path.join(self._base_dir, val_file_dir)
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.strip() for item in self.sample_list if item.strip()]
        else:
            print(f"Warning: Split file {file_path} not found.")

        print("total {}  {} samples".format(len(self.sample_list), self.mode))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        # CBD dataset structure:
        # images/Species/Image.jpg
        # masks/Species/Image.png
        
        # Handle case where extension is included in the list (e.g. from butterfly_200 train.txt)
        case_name, case_ext = os.path.splitext(case)
        if case_ext:
            # If extension exists in list, try to respect it or find alternative if missing
            # But usually we treat 'case' as the ID.
            pass
        
        image_path = None
        # Try direct path (if case has extension)
        p = os.path.join(self._base_dir, 'images', case)
        if os.path.exists(p):
            image_path = p
        else:
            # Try images_small (for butterfly_200)
            p_small = os.path.join(self._base_dir, 'images_small', case)
            if os.path.exists(p_small):
                image_path = p_small
            else:
                 # Try appending extensions
                 for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                    # Try images folder
                    p = os.path.join(self._base_dir, 'images', case + ext)
                    if os.path.exists(p):
                        image_path = p
                        break
                    # Try images_small folder
                    p_small = os.path.join(self._base_dir, 'images_small', case + ext)
                    if os.path.exists(p_small):
                        image_path = p_small
                        break
                    
                    # If case had extension (e.g. .png) but file is .jpg
                    if case_ext:
                        # Try replacing extension
                        p_replace = os.path.join(self._base_dir, 'images', case_name + ext)
                        if os.path.exists(p_replace):
                            image_path = p_replace
                            break
                        p_small_replace = os.path.join(self._base_dir, 'images_small', case_name + ext)
                        if os.path.exists(p_small_replace):
                            image_path = p_small_replace
                            break

        if image_path is None:
             # Fallback
             print(f"Image not found for case: {case}")
             # Assume images/case.jpg as default fallback
             image_path = os.path.join(self._base_dir, 'images', case_name + '.jpg')

        # Find mask
        mask_path = None
        
        # Determine mask directory (masks or SegmentationClass)
        mask_dirs = ['masks', 'SegmentationClass']
        
        for mask_dir in mask_dirs:
            # Try masks folder
            p_mask = os.path.join(self._base_dir, mask_dir, case)
            if os.path.exists(p_mask):
                mask_path = p_mask
                break
            else:
                # Try appending extensions
                for ext in ['.png', '.jpg', '.bmp']:
                    p = os.path.join(self._base_dir, mask_dir, case + ext)
                    if os.path.exists(p):
                        mask_path = p
                        break
                    if case_ext:
                        p_replace = os.path.join(self._base_dir, mask_dir, case_name + ext)
                        if os.path.exists(p_replace):
                            mask_path = p_replace
                            break
                if mask_path:
                    break
        
        if mask_path is None:
             # Fallback
             mask_path = os.path.join(self._base_dir, 'masks', case_name + '.png')

        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
             # print(f"Failed to read image: {image_path}")
             # Create dummy image to avoid crash during dev/debug if file missing
             image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB

        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
             # print(f"Failed to read mask: {mask_path}")
             label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Ensure image and mask have the same size
        if image.shape[:2] != label.shape[:2]:
            label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        
        label = label[..., None] # Add channel dim

        # Manual normalization/transpose if not done by transform
        if not isinstance(image, (np.ndarray, np.generic)):
             # Assume tensor
             pass
        else:
            image = image.astype('float32')
            image = image.transpose(2, 0, 1) / 255
            
            label = label.astype('float32') 
            label = label.transpose(2, 0, 1) / 255
            label[label > 0] = 1

        sample = {"image": image, "label": label, "case": case}
        return sample
