"""
    Fine-tune / Train a vision model on a dataset given.
"""

# Use: python3 train_vision_model_sim.py
#   --weights <weight_dir>
#   --dataset <dataset_dir>

from ultralytics import YOLO
import argparse
from pathlib import Path
import torch

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="",
                        help="vision model weights to train/fine tune on")
    parser.add_argument('--dataset', type=str, default="",
                        help="dataset YAML to train model on")
    parser.add_argument('--freeze', type=int, default=0,
                        help="whether to freeze backbone")
    return parser

def main():
    args = parser().parse_args()
    
    model = YOLO(args.weights)
    model.train(
        data=args.dataset,
        epochs=200,
        patience=20,
        imgsz=640,
        freeze=10 if args.freeze else 0, # locks backbone
        lr0=0.001,
        device=("cuda" if torch.cuda.is_available() else "cpu")
    )

if __name__ == "__main__":
    main()