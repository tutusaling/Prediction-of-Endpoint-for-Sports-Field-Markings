import os
import json
import cv2 as cv
import numpy as np
from utils import save_mask
from constants import class2id_1b, id2id_sym_1b
from tqdm import tqdm


# create the masks for the dataset
def create_masks(dataset_dir, mask_dir):
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir, exist_ok=True)
    
    frames = [f for f in os.listdir(dataset_dir) if ".jpg" in f]
    frames = sorted(frames)

    print(f"Creating masks for {dataset_dir}...")

    for frame in tqdm(frames):
        frame_index = frame.split(".")[0]
        
        # get json file
        annotation_file = os.path.join(dataset_dir, f"{frame_index}.json")
        if not os.path.exists(annotation_file):
            continue
        with open(annotation_file, "r") as f:
            groundtruth_lines = json.load(f)
        
        # get image and shape
        img_path = os.path.join(dataset_dir, frame)
        img = cv.imread(img_path)
        height, width, _ = img.shape

        # create mask 0 is the background
        mask = np.zeros((height, width), dtype=np.uint8)
        for class_, class_number in class2id_1b.items():
            if class_ in groundtruth_lines.keys():
                key = class_
                line = groundtruth_lines[key]
                prev_point = line[0]
                for i in range(1, len(line)):
                    next_point = line[i]
                    cv.line(mask,
                            (int(prev_point["x"] * width), int(prev_point["y"] * height)),
                            (int(next_point["x"] * width), int(next_point["y"] * height)),
                            class_number,
                            2)
                    prev_point = next_point


        mask_path = os.path.join(mask_dir, f"{frame_index}.png")
        save_mask(mask, mask_path)

        # create hflip mask
        mask = cv.flip(mask, 1)
        updated_mask = np.copy(mask)
        for class_number, class_sym_number in id2id_sym_1b.items():
            updated_mask[mask == class_number] = class_sym_number
        mask = updated_mask

        mask_path = os.path.join(mask_dir, f"{frame_index}_hflip.png")
        save_mask(mask, mask_path)

    print(f"Masks have been saved in {mask_dir}")


if __name__ == "__main__":
    
    # create masks for the dataset
    dataset_dirs = ["../SoccerNet/train", "../SoccerNet/valid", "../SoccerNet/test"]
    mask_dirs = ["../SoccerNet/mask/train", "../SoccerNet/mask/valid", "../SoccerNet/mask/test"]

    for dataset_dir, mask_dir in zip(dataset_dirs, mask_dirs):
        if not os.path.exists(dataset_dir):
            continue

        create_masks(dataset_dir, mask_dir)
