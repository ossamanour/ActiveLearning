import argparse
from ultralytics import YOLO

def train_model(data_yaml, model_path, epochs=50, imgsz=640, batch=16):
    model = YOLO(model_path)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        exist_ok=True
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    
    print(f"\nStarting training on {args.data} with model {args.model}")
    train_model(args.data, args.model, args.epochs, args.imgsz, args.batch)
    print("Training completed!")