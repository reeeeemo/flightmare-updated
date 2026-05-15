from ultralytics import YOLO
import argparse
from pathlib import Path
import torch

# Fine-tunes / Trains a vision model on a dataset given via simulation.

# Use: python3 train_vision_model_sim.py
#   --weights <weight_dir>
#   --dataset <dataset_dir>

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="",
                        help="vision model weights to train/fine tune on")
    parser.add_argument('--dataset', type=str, default="",
                        help="dataset YAML to train model on")
    return parser

def main():
    args = parser().parse_args()
    
    model = YOLO(args.weights)
    model.train(
        data=args.dataset,
        epochs=200,
        patience=20,
        imgsz=384,
        freeze=10, # locks backbone
        lr0=0.001,
        device=("cuda" if torch.cuda.is_available() else "cpu")
    )

if __name__ == "__main__":
    main()