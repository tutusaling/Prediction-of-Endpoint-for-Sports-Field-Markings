
import os
import random

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from transformers import SegformerImageProcessor
from lightning import LightningDataModule

from constants import MEAN, STD, BAD_FRAMES


class SoccerNetDataset(Dataset):
    def __init__(self,
                 datasetpath,
                 maskpath,
                 split="test",
                 size=(512, 512),
                 skip_bad_frames=False,
                 as_train=False,
                 do_normalize=True,
                 ):
        
        # Check if the dataset path exists
        dataset_dir = os.path.join(datasetpath, split)
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Invalid dataset path: {dataset_dir}")
        
        # Define the mean and std of the dataset
        self.mean = MEAN
        self.std = STD
        self.size = size
        self.as_train = as_train
        self.split = split
        
        # Get the list of images
        frames = [f for f in os.listdir(dataset_dir) if ".jpg" in f]
        
        # Remove bad frames if needed
        frames = sorted(frames)
        if skip_bad_frames:
            frames = [f for i, f in enumerate(frames) if i not in BAD_FRAMES[self.split]]

        # Create the data list containing the images path and the annotations json
        self.data = []
        for frame in frames:
            frame_index = frame.split(".")[0]
            annotation_file = os.path.join(dataset_dir, f"{frame_index}.json")
            mask_file = os.path.join(maskpath, split, f"{frame_index}.png")
            mask_flip_file = os.path.join(maskpath, split, f"{frame_index}_hflip.png")
            
            # make sure the all files exists
            if not os.path.exists(mask_file):
                continue
            if not os.path.exists(mask_flip_file):
                continue
            if not os.path.exists(annotation_file):
                continue
            
            img_path = os.path.join(dataset_dir, frame)
            
            self.data.append({
                "image_path": img_path,
                "mask_path": mask_file,
                "mask_flip_path": mask_flip_file,
            })

        # train transforms
        self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

        self.img_processor = SegformerImageProcessor(do_reduce_labels=True, do_normalize=do_normalize, image_mean=self.mean, image_std=self.std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        image = Image.open(item["image_path"]).resize(self.size)
        if self.as_train:
            image = self.color_jitter(image)
            if random.random() > 0.5:
                mask = Image.open(item["mask_path"]).resize(self.size, resample=Image.NEAREST)
            else:
                mask = Image.open(item["mask_flip_path"]).resize(self.size, resample=Image.NEAREST)
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            mask = Image.open(item["mask_path"]).resize(self.size, resample=Image.NEAREST)

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.img_processor(image, mask, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs
    


class SoccerNetDataModule(LightningDataModule):
    def __init__(self, 
                 root,
                 mask_root, 
                 train_folder,
                 valid_folder,
                 test_folder,
                 batch_size=6,
                 num_workers=4
                 ):
        
        super().__init__()
        # Define the dataset path
        self.root = root
        self.mask_root = mask_root
        
        # Define the train, valid and test folders
        self.train_folder = train_folder
        self.valid_folder = valid_folder
        self.test_folder = test_folder
        
        # Define the batch size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define the datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage=None):
        self.train_dataset = SoccerNetDataset(self.root, self.mask_root, split=self.train_folder, as_train=True)
        self.val_dataset = SoccerNetDataset(self.root, self.mask_root, split=self.valid_folder)
        self.test_dataset = SoccerNetDataset(self.root, self.mask_root, split=self.test_folder)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)