import argparse
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from collections import defaultdict

# python convert_coco_to_yolo.py \
#     --coco_json annotations/instances_train2017.json \
#     --image_dir train2017 \
#     --classes person car dog \
#     --output_root ./my_yolo_dataset

# COCO class name to ID mapping (80 classes)
COCO_CLASSES = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "fire hydrant": 11, "stop sign": 12, "parking meter": 13, "bench": 14,
    "bird": 15, "cat": 16, "dog": 17, "horse": 18, "sheep": 19, "cow": 20,
    # ... (add all 80 classes)
}

def parse_args():
    parser = argparse.ArgumentParser(description='Convert COCO to YOLO format for specific classes')
    parser.add_argument('--coco_json', type=str, required=True, help='Path to COCO JSON annotations')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing COCO images')
    parser.add_argument('--output_root', type=str, default='./yolo_dataset', help='Root output directory')
    parser.add_argument('--classes', nargs='+', required=True, 
                       help='List of class names (e.g., "person car dog")')
    return parser.parse_args()

def validate_classes(class_names):
    """Convert class names to COCO IDs and validate"""
    class_ids = []
    for name in class_names:
        if name.lower() not in COCO_CLASSES:
            raise ValueError(f"Class '{name}' not found in COCO dataset. Valid classes: {list(COCO_CLASSES.keys())}")
        class_ids.append(COCO_CLASSES[name.lower()])
    return class_ids

def main():
    args = parse_args()
    
    # Validate classes and get IDs
    class_ids = validate_classes(args.classes)
    yolo_class_map = {coco_id: idx for idx, coco_id in enumerate(class_ids)}  # Maps COCO ID → YOLO ID (0-based)

    # Create output directories
    os.makedirs(args.output_root, exist_ok=True)
    
    # Load COCO annotations
    try:
        coco = COCO(args.coco_json)
    except Exception as e:
        raise SystemExit(f"❌ Failed to load COCO annotations: {str(e)}")

    # Process each class
    for class_name, coco_id in zip(args.classes, class_ids):
        print(f"\nProcessing class: {class_name} (COCO ID: {coco_id})")
        
        # Create class-specific directories
        class_image_dir = os.path.join(args.output_root, f"{class_name}_images")
        class_label_dir = os.path.join(args.output_root, f"{class_name}_labels")
        os.makedirs(class_image_dir, exist_ok=True)
        os.makedirs(class_label_dir, exist_ok=True)

        # Get all annotations for this class
        ann_ids = coco.getAnnIds(catIds=[coco_id])
        annotations = coco.loadAnns(ann_ids)
        
        # Group annotations by image
        img_to_anns = defaultdict(list)
        for ann in annotations:
            img_to_anns[ann['image_id']].append(ann)

        # Process each image
        for img_id, anns in tqdm(img_to_anns.items(), desc=f"Processing {class_name}"):
            try:
                img_info = coco.loadImgs(img_id)[0]
                src_img_path = os.path.join(args.image_dir, img_info['file_name'])
                
                # Skip if image doesn't exist
                if not os.path.exists(src_img_path):
                    continue

                # Get image dimensions
                img = cv2.imread(src_img_path)
                if img is None:
                    continue
                    
                h, w = img.shape[:2]

                # Create YOLO label file
                base_name = os.path.splitext(img_info['file_name'])[0]
                label_file = os.path.join(class_label_dir, f"{base_name}.txt")
                
                with open(label_file, 'w') as f:
                    for ann in anns:
                        try:
                            mask = coco.annToMask(ann)
                            ys, xs = np.where(mask > 0)
                            
                            if len(xs) == 0:
                                continue
                                
                            x_min, x_max = np.min(xs), np.max(xs)
                            y_min, y_max = np.min(ys), np.max(ys)

                            # Convert to YOLO format
                            x_center = ((x_min + x_max) / 2) / w
                            y_center = ((y_min + y_max) / 2) / h
                            width = (x_max - x_min) / w
                            height = (y_max - y_min) / h

                            f.write(f"{yolo_class_map[coco_id]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        
                        except Exception as e:
                            continue

                # Copy image to class directory
                dst_img_path = os.path.join(class_image_dir, img_info['file_name'])
                shutil.copy2(src_img_path, dst_img_path)

            except Exception as e:
                continue

    print("\n✅ Conversion complete!")
    print(f"Dataset structure at: {os.path.abspath(args.output_root)}")
    print("Class directories created:")
    for class_name in args.classes:
        print(f"  - {class_name}_images/")
        print(f"  - {class_name}_labels/")

if __name__ == "__main__":
    main()