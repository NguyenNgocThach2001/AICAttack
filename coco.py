"""Official processing for COCO and Flickr8k datasets."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys
import os
from scipy.misc import imread, imresize
import json
from PIL import Image
from pathlib import Path

# ===== DIRECTORY CONSTANTS =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

# COCO Dataset paths
COCO_TRAIN_IMAGES_DIR = DATA_DIR / 'coco'/ 'images'
COCO_TRAIN_CAPTIONS_JSON = DATA_DIR /'coco'/ 'annotations' / 'captions_train2014.json'

# Flickr8k Dataset paths
FLICKR8K_IMAGE_DIR = DATA_DIR / 'flickr8k' / 'Images'
FLICKR8K_CAPTIONS_FILE = DATA_DIR / 'flickr8k' / 'captions.txt'
# ===== IMAGE SIZE CONSTANTS =====
COCO_RESIZE_SIZE = (255, 255)
FLICKR8K_RESIZE_SIZE = (255, 255)
FLICKR8K_LARGE_RESIZE_SIZE = (384, 384)

# ===== CAPTION PARSING CONSTANTS =====
CAPTION_DELIMITER = '\t'
IMAGE_ID_SEPARATOR = '#'


def ensure_dir_exists(directory: Path) -> Path:
    """
    Ensure that a directory exists, create it if it doesn't.
    
    Args:
        directory: Path object representing the directory
        
    Returns:
        Path object of the created/existing directory
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# Ensure data directories exist
ensure_dir_exists(DATA_DIR)
ensure_dir_exists(COCO_TRAIN_IMAGES_DIR.parent)


# Initialize COCO dataset
cap = dset.CocoCaptions(
    root=str(COCO_TRAIN_IMAGES_DIR),
    annFile=str(COCO_TRAIN_CAPTIONS_JSON),
    transform=transforms.Compose([
        transforms.Resize(size=COCO_RESIZE_SIZE)
    ])
)


def get_coco_dataset(id=3, for_attention=False):
    """
    Load image and captions from COCO dataset.
    
    Args:
        id: Image index in dataset
        for_attention: If True, convert image to tensor
        
    Returns:
        If for_attention is True: image tensor
        Otherwise: (dataset object, captions, PIL image)
    """
    img, target = cap[id]
    
    if for_attention:
        transform_totensor = transforms.ToTensor()
        img = transform_totensor(img)
        return img
    else:
        return cap, target, img


def get_coco_dataset_for_sat(id, for_attention=False):
    """
    Load COCO image for SAT (Show, Attend and Tell) model.
    
    Args:
        id: Image index in dataset
        for_attention: Not used, kept for compatibility
        
    Returns:
        Tuple of (dataset object, captions, PIL image)
    """
    pil_image, target = cap[id][0], cap[id][1]
    return cap, target, pil_image


def parse_flickr8k_captions():
    """
    Parse Flickr8k captions file into a dictionary.
    
    Returns:
        Dictionary mapping image names to lists of captions
    """
    captions_file = FLICKR8K_CAPTIONS_FILE
    
    with open(captions_file, 'r') as f:
        captions_data = f.readlines()

    captions_dict = {}
    for line in captions_data:
        parts = line.strip().split(CAPTION_DELIMITER)
        if len(parts) < 2:
            continue
            
        image_name = parts[0].split(IMAGE_ID_SEPARATOR)[0]
        caption = parts[1]
        
        if image_name not in captions_dict:
            captions_dict[image_name] = []
        captions_dict[image_name].append(caption)

    return captions_dict


def get_flickr8k_dataset_for_sat(id):
    """
    Load Flickr8k image and captions for SAT model.
    
    Args:
        id: Image index (0-based)
        
    Returns:
        Tuple of (captions list, captions list, image tensor)
        or (empty list, None) if invalid ID
    """
    captions_dict = parse_flickr8k_captions()
    image_ids = list(captions_dict.keys())

    if 0 <= id < len(image_ids):
        image_id = image_ids[id]
        captions = captions_dict[image_id]
        
        # Load and resize image
        image_file = FLICKR8K_IMAGE_DIR / image_id
        image = Image.open(image_file)
        image_resized = image.resize(FLICKR8K_RESIZE_SIZE, Image.ANTIALIAS)
        
        transform = transforms.Compose([transforms.ToTensor()])
        image_data = transform(image_resized)
        
        return captions, captions, image_data
    else:
        print(f"Invalid image number: {id}. Valid range: 0-{len(image_ids)-1}")
        return [], None


def get_flickr8k_dataset(id):
    """
    Load Flickr8k image and captions with larger resize.
    
    Args:
        id: Image index (0-based)
        
    Returns:
        Tuple of (captions list, captions list, image tensor)
        or (empty list, None) if invalid ID
    """
    captions_dict = parse_flickr8k_captions()
    image_ids = list(captions_dict.keys())

    if 0 <= id < len(image_ids):
        image_id = image_ids[id]
        captions = captions_dict[image_id]
        
        # Load and resize image
        image_file = FLICKR8K_IMAGE_DIR / image_id
        image = Image.open(image_file)
        image_resized = image.resize(FLICKR8K_LARGE_RESIZE_SIZE, Image.ANTIALIAS)
        
        transform = transforms.Compose([transforms.ToTensor()])
        image_data = transform(image_resized)
        
        return captions, captions, image_data
    else:
        print(f"Invalid image number: {id}. Valid range: 0-{len(image_ids)-1}")
        return [], None


def get_flickr8k_image_count():
    """
    Get the total number of images in Flickr8k dataset.
    
    Returns:
        Integer count of images
    """
    captions_dict = parse_flickr8k_captions()
    return len(captions_dict)


def get_coco_image_count():
    """
    Get the total number of images in COCO dataset.
    
    Returns:
        Integer count of images
    """
    return len(cap)


if __name__ == '__main__':
    print("COCO dataset info:")
    print(f"Total images: {get_coco_image_count()}")
    print(f"Sample image: {get_coco_dataset_for_sat(0)[2]}")
    
    print("\nFlickr8k dataset info:")
    print(f"Total images: {get_flickr8k_image_count()}")
    captions, _, img = get_flickr8k_dataset(5)
    print(f"Sample captions: {captions[:2]}")