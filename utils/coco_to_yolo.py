from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from collections import defaultdict

# Configuration
COCO_JSON_PATH = "data/coco/annotations/instances_val2017.json"
IMAGE_DIR = "data/coco/images/val2017"
OUTPUT_LABEL_DIR = "val_horse_labels"
OUTPUT_IMAGE_DIR = "val_horse_images"  # New folder for horse images
HORSE_CLASS_ID = 19  # COCO class ID for horses
YOLO_CLASS_ID = 0    # Class ID in YOLO format

def main():
    # Create output directories
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    # Load COCO annotations
    try:
        coco = COCO(COCO_JSON_PATH)
    except Exception as e:
        print(f"❌ Failed to load COCO annotations: {str(e)}")
        return

    # Get all horse annotations
    horse_ann_ids = coco.getAnnIds(catIds=[HORSE_CLASS_ID])
    horse_annotations = coco.loadAnns(horse_ann_ids)
    
    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in horse_annotations:
        img_to_anns[ann['image_id']].append(ann)

    # Process each image containing horses
    for img_id, anns in tqdm(img_to_anns.items(), desc="Processing horse images"):
        try:
            img_info = coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            src_img_path = os.path.join(IMAGE_DIR, img_filename)
            dst_img_path = os.path.join(OUTPUT_IMAGE_DIR, img_filename)
            
            # Verify source image exists
            if not os.path.exists(src_img_path):
                print(f"⚠️ Image not found: {src_img_path}")
                continue

            # Get image dimensions
            img = cv2.imread(src_img_path)
            if img is None:
                print(f"⚠️ Could not read image: {src_img_path}")
                continue
                
            h, w = img.shape[:2]

            # Create YOLO label file
            label_file = os.path.join(OUTPUT_LABEL_DIR, os.path.splitext(img_filename)[0] + '.txt')
            with open(label_file, 'w') as f:
                for ann in anns:
                    try:
                        mask = coco.annToMask(ann)
                        ys, xs = np.where(mask > 0)
                        
                        if len(xs) == 0:
                            continue
                            
                        x_min, x_max = np.min(xs), np.max(xs)
                        y_min, y_max = np.min(ys), np.max(ys)

                        x_center = ((x_min + x_max) / 2) / w
                        y_center = ((y_min + y_max) / 2) / h
                        width = (x_max - x_min) / w
                        height = (y_max - y_min) / h

                        f.write(f"{YOLO_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    except Exception as ann_error:
                        print(f"⚠️ Annotation error in {img_filename}: {str(ann_error)}")
                        continue

            # Move the image to horse_images folder
            shutil.copy2(src_img_path, dst_img_path)  # copy2 preserves metadata

        except Exception as img_error:
            print(f"⚠️ Image processing error (ID {img_id}): {str(img_error)}")
            continue

    print(f"\n✅ Conversion complete!")
    print(f"YOLO labels saved to: {os.path.abspath(OUTPUT_LABEL_DIR)}")
    print(f"Horse images saved to: {os.path.abspath(OUTPUT_IMAGE_DIR)}")

if __name__ == "__main__":
    main()