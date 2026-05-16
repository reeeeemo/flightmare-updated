import cv2
import argparse
from pathlib import Path


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="",
                        help="dataset YAML to train model on")
    return parser

def main():
    args = parser().parse_args()
    
    dataset_dir = Path(args.dataset)
    if dataset_dir.exists():
        # get all images and lbls given they are same size folders
        lbls_dir = dataset_dir / "labels" / "train"
        imgs_dir = dataset_dir / "images" / "train"
        all_images = list(p for p in imgs_dir.iterdir() if p.is_file())
        images_by_stem = {f.stem: f for f in all_images}
        all_labels = list(p for p in lbls_dir.iterdir() if p.is_file())
        
        for lbl in all_labels:
            img = cv2.imread(images_by_stem[lbl.stem])
            h,w = img.shape[:2]
    
            with open(lbl, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    values = line.strip().split(' ')
                    cx, cy, bw, bh = [float(v) for v in values[1:5]]
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                        
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    for j in range(4):
                        u = int(float(values[5+j*3]) * w) 
                        v = int(float(values[5+j*3+1]) * h)
                        cv2.circle(img, (u,v), 5, (0,0,255),-1)

            cv2.imshow('label_check', img)
            cv2.waitKey(0)
    else:
        raise RuntimeError("Dataset does not exist.")
    
if __name__ == "__main__":
    main()