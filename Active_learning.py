import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import shutil


# python active_learning.py \
#     --dataset_dir ./coco_data \
#     --output_dir ./selected_horses \
#     --target_class horse \
#     --model_path ./models/yolov8x.pt \
#     --selected_ratio 0.7 \
#     --detection_threshold 0.15


# COCO class names and IDs (example: "horse" is class 17)
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

def calculate_entropy(probabilities):
    """Calculate entropy for a set of probabilities."""
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def compute_informativeness_score(image_path, model, target_class_id, detection_threshold=0.1):
    """Compute score based only on target class detections."""
    image = cv2.imread(image_path)
    results = model(image, conf=detection_threshold, verbose=False)
    
    informativeness = 0.0
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            if cls_id == target_class_id:
                cls_prob = box.conf.cpu().numpy()
                entropy = calculate_entropy(np.array([cls_prob, 1 - cls_prob]))
                informativeness += entropy
    return informativeness

def rank_images_by_informativeness(images_dir, model, target_class_id, detection_threshold):
    """Rank images by target class uncertainty."""
    image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir) 
                  if img.endswith(('.jpg', '.png', '.jpeg'))]
    return sorted(
        [(img_path, compute_informativeness_score(img_path, model, target_class_id, detection_threshold)) 
         for img_path in tqdm(image_paths, desc="Scoring images")],
        key=lambda x: x[1], reverse=True
    )

def save_selected_data(ranked_images, labels_dir, output_dir, selected_ratio):
    """Save top N% images and their labels."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    num_selected = int(len(ranked_images) * selected_ratio)
    for img_path, _ in tqdm(ranked_images[:num_selected], desc="Saving data"):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        
        if os.path.exists(label_path):
            shutil.copy(img_path, os.path.join(output_dir, "images", img_name))
            shutil.copy(label_path, os.path.join(output_dir, "labels", label_name))

def active_learning_for_class(dataset_dir, output_dir, target_class_name, model_path, selected_ratio=0.8, detection_threshold=0.1):
    """Main Active Learning pipeline."""
    # Validate target class
    target_class_id = next((k for k, v in COCO_CLASSES.items() if v == target_class_name.lower()), None)
    if target_class_id is None:
        raise ValueError(f"Class '{target_class_name}' not found. Valid options: {list(COCO_CLASSES.values())}")
    
    # Load model
    model = YOLO(model_path)
    
    # Process dataset
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    ranked_images = rank_images_by_informativeness(images_dir, model, target_class_id, detection_threshold)
    save_selected_data(ranked_images, labels_dir, output_dir, selected_ratio)
    
    print(f"\nSelected {int(len(ranked_images)*selected_ratio)} images with uncertain '{target_class_name}' detections")
    print(f"Results saved to: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Active Learning for Specific COCO Class")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                       help="Path to dataset (must contain 'images' and 'labels' folders)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for selected samples")
    parser.add_argument("--target_class", type=str, required=True,
                       help="COCO class name (e.g., 'horse', 'person')")
    parser.add_argument("--model_path", type=str, default="yolov8n.pt",
                       help="Path to YOLO model (default: yolov8n.pt)")
    parser.add_argument("--selected_ratio", type=float, default=0.8,
                       help="Ratio of samples to select (default: 0.8)")
    parser.add_argument("--detection_threshold", type=float, default=0.1,
                       help="Detection confidence threshold (default: 0.1)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate paths
    dataset_dir = Path(args.dataset_dir)
    if not (dataset_dir/"images").exists():
        raise FileNotFoundError(f"Missing 'images' folder in {dataset_dir}")
    if not (dataset_dir/"labels").exists():
        raise FileNotFoundError(f"Missing 'labels' folder in {dataset_dir}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    active_learning_for_class(
        dataset_dir=str(dataset_dir),
        output_dir=args.output_dir,
        target_class_name=args.target_class,
        model_path=args.model_path,
        selected_ratio=args.selected_ratio,
        detection_threshold=args.detection_threshold
    )

if __name__ == "__main__":
    main()