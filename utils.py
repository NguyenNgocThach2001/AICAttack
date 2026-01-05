import torch
import os
from pathlib import Path
import pickle
from PIL import Image, ImageDraw
import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import requests
from coco import *

# ===== DEVICE CONFIGURATION =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== DIRECTORY CONSTANTS =====
OUTPUT_DIR = Path('outputs')
OUT_BLIP_FLICKER8K = OUTPUT_DIR / 'blip_flicker8k_1000p..._overall_384_attacked'
OUT_DEMO_PAPER = OUTPUT_DIR / 'demo_paper'
OUT_SEPARATE_TEST = OUTPUT_DIR / 'separate_test_500p_lm'
OUT_SEPARATE_COMBINED = OUTPUT_DIR / 'separate_combined_img'

# ===== IMAGE SAVING CONSTANTS =====
BLIP_ATTACK_OUTPUT_DIR = OUTPUT_DIR / 'blip_flicker8k_1000p_100samples_overall_384_attacked'
ALPHAS_FILE = Path('alphas.txt')

# ===== IMAGE SIZE CONSTANTS =====
DEFAULT_IMAGE_SIZE = 384
SAT_IMAGE_SIZE = 255

# ===== NORMALIZATION CONSTANTS =====
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def load_image(id, image_size, device, before=True, dataset='coco'):
    """
    Load and transform an image from specified dataset.
    
    Args:
        id: Image ID/index
        image_size: Target size for image resize
        device: Torch device (cuda/cpu)
        before: If True, return tensor in [0,255] range, else [-1,1] normalized
        dataset: Dataset name ('coco' or 'flicker8k')
        
    Returns:
        Transformed image tensor
    """
    if dataset == 'coco':
        _, _, raw_image = get_coco_dataset(id)
    elif dataset == 'flicker8k':
        _, _, raw_image = get_flickr8k_dataset(id)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Transform image to tensor [0, 255]
    transform_pil_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.PILToTensor(),
    ])

    # Transform image to tensor [-1, 1]
    transform_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    if before:
        image = transform_pil_tensor(raw_image).to(device)
    else:
        image = transform_tensor(raw_image).unsqueeze(0).to(device)

    return image


def load_image_for_sat(id, image_size, device, before=True, dataset='flicker8k'):
    """
    Load image for SAT (Show, Attend and Tell) model.
    
    Args:
        id: Image ID/index
        image_size: Target size for image resize
        device: Torch device (cuda/cpu)
        before: If True, return raw tensor, else normalized
        dataset: Dataset name ('coco' or 'flicker8k')
        
    Returns:
        Transformed image tensor
    """
    if dataset == 'coco':
        folder_path = COCO_TRAIN_IMAGES_DIR
        files = sorted(os.listdir(folder_path))
        img = imread(str(folder_path / files[id]))
        
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        
        img = imresize(img, (SAT_IMAGE_SIZE, SAT_IMAGE_SIZE))
        img = img.transpose(2, 0, 1)
        
        if not before:
            img = img / 255.
            img = torch.FloatTensor(img).to(device)
            normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            transform = transforms.Compose([normalize])
            img = transform(img)
        else:
            img = torch.from_numpy(img)
            
    elif dataset == 'flicker8k':
        captions_file = FLICKR8K_CAPTIONS_FILE
        image_dir = FLICKR8K_IMAGE_DIR
        
        with open(captions_file, 'r') as f:
            captions_data = f.readlines()

        captions_dict = {}
        for line in captions_data:
            parts = line.strip().split('\t')
            image_name = parts[0].split('#')[0]
            caption = parts[1]
            if image_name not in captions_dict:
                captions_dict[image_name] = []
            captions_dict[image_name].append(caption)

        image_ids = list(captions_dict.keys())

        if 0 <= id < len(image_ids):
            image_id = image_ids[id]
            image_file = os.path.join(image_dir, image_id)
            image = Image.open(image_file)
            image_resized = image.resize((SAT_IMAGE_SIZE, SAT_IMAGE_SIZE), Image.ANTIALIAS)
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(image_resized)
            
            if not before:
                img = img / 255.
                img = torch.FloatTensor(img).to(device)
                normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                transform = transforms.Compose([normalize])
                img = transform(img)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return img


def divide_list(lst, num_chunks=5):
    """
    Divide a list into approximately equal chunks.
    
    Args:
        lst: List to divide
        num_chunks: Number of chunks to create
        
    Returns:
        List of chunks
    """
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    result = []
    start = 0
    
    for i in range(num_chunks):
        end = start + chunk_size
        if remainder > 0:
            end += 1
            remainder -= 1
        result.append(lst[start:end])
        start = end
        
    return result


def save_image(id, pixels, figure, is_attention='att'):
    """
    Save a perturbed image to disk.
    
    Args:
        id: Image ID
        pixels: Number of perturbed pixels
        figure: Image tensor to save
        is_attention: Label for attention mechanism used
        
    Returns:
        Status string
    """
    ensure_dir_exists(BLIP_ATTACK_OUTPUT_DIR)
    
    transform_toimage = transforms.ToPILImage()
    test_figure = transform_toimage(figure)
    output_path = BLIP_ATTACK_OUTPUT_DIR / f'{id}_{pixels}_{is_attention}.jpg'
    test_figure.save(str(output_path))

    return 'saved'


def save_raw_image(id):
    """
    Save raw image for demo/paper purposes.
    
    Args:
        id: Image ID
        
    Returns:
        Status string
    """
    ensure_dir_exists(OUT_DEMO_PAPER)
    
    image_384 = dataloader.get_item('coco', id, DEFAULT_IMAGE_SIZE)[1]
    transform_toimage = transforms.ToPILImage()
    test_figure = transform_toimage(image_384)
    output_path = OUT_DEMO_PAPER / f"{id}_raw_image.jpg"
    test_figure.save(str(output_path))
    
    return 'saved'

    
def convert_to_preferred_format(sec):
    """
    Convert seconds to HH:MM:SS format.
    
    Args:
        sec: Number of seconds
        
    Returns:
        Formatted time string
    """
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


def load_json(dir):
    """
    Load data from a pickle file.
    
    Args:
        dir: File path
        
    Returns:
        Loaded data
    """
    with open(dir, 'rb') as file:
        data = pickle.load(file)
    return data


def image_path(id, dataset='coco'):
    """
    Get the file path for a specific image.
    
    Args:
        id: Image ID/index
        dataset: Dataset name ('coco' or 'flicker8k')
        
    Returns:
        String path to image file
    """
    if dataset == 'coco':
        folder_path = COCO_TRAIN_IMAGES_DIR
        files = sorted(os.listdir(folder_path))
        return str(folder_path / files[id])
        
    elif dataset == 'flicker8k':
        image_dir = FLICKR8K_IMAGE_DIR
        image_filenames = sorted(os.listdir(image_dir))
        if 0 <= id < len(image_filenames):
            image_filename = image_filenames[id]
            return str(image_dir / image_filename)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def filter_alphas():
    """
    Process and save alpha values to file (legacy function).
    """
    output_array = output_tensor.numpy()

    with open(ALPHAS_FILE, 'w') as file:
        for row in output_array:
            for element in row:
                file.write(str(element) + ' ')
            file.write('\n')

    output_data = []
    with open(ALPHAS_FILE, 'r') as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split(' ')
            row = [float(element) for element in elements if element]
            output_data.append(row)
            
    output_tensor = torch.tensor(output_data)
    print(output_tensor)


def circle_area(image_path, pixel_coordinates, radius=10, outline_color=(255, 0, 0), outline_width=2):
    """
    Draw circles around specified pixel coordinates on an image.
    
    Args:
        image_path: Path to input image
        pixel_coordinates: List of (x, y) tuples
        radius: Circle radius (not used in current implementation)
        outline_color: RGB tuple for circle color
        outline_width: Width of circle outline
        
    Returns:
        PIL Image with circles drawn
    """
    image = Image.open(image_path)
    circled_image = Image.new("RGBA", image.size)
    circled_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(circled_image)

    min_x = min(pixel_coordinates, key=lambda coord: coord[0])[0]
    min_y = min(pixel_coordinates, key=lambda coord: coord[1])[1]
    max_x = max(pixel_coordinates, key=lambda coord: coord[0])[0]
    max_y = max(pixel_coordinates, key=lambda coord: coord[1])[1]
    bbox = (min_x, min_y, max_x, max_y)

    draw.ellipse(bbox, outline=outline_color, width=outline_width)

    return circled_image


def draw_multi_image(id):
    """
    Combine multiple attack result images into a single visualization.
    
    Args:
        id: Image ID
    """
    ensure_dir_exists(OUT_SEPARATE_COMBINED)
    
    # Define image paths
    image_paths = {
        'separate': OUT_SEPARATE_TEST / f"{id}_separate.png",
        'att_allin': OUT_SEPARATE_TEST / f"{id}_att_allin.png",
        'att_sep': OUT_SEPARATE_TEST / f"{id}_att_sep.png",
        'noatt': OUT_SEPARATE_TEST / f"{id}_noatt.png",
        'allin': OUT_SEPARATE_TEST / f"{id}_allin.png"
    }
    
    # Load images
    images = {key: Image.open(path) for key, path in image_paths.items()}
    
    # Calculate combined image dimensions
    width = max(
        images['separate'].width + images['att_allin'].width,
        images['att_sep'].width + images['noatt'].width
    )
    height = (images['separate'].height + images['att_sep'].height + 
              max(images['att_allin'].height, images['noatt'].height) + 
              images['allin'].height)

    # Create combined image
    combined_image = Image.new('RGB', (width, height))

    # Paste images
    combined_image.paste(images['separate'], (0, 0))
    combined_image.paste(images['att_allin'], (images['separate'].width, 0))
    combined_image.paste(images['att_sep'], (0, images['separate'].height))
    combined_image.paste(images['noatt'], (images['att_sep'].width, images['att_allin'].height))
    combined_image.paste(images['allin'], 
                        ((width - images['allin'].width) // 2, 
                         images['separate'].height + images['att_sep'].height))

    # Trim excess white space
    combined_image = combined_image.crop(combined_image.getbbox())
    
    # Create plot
    fig, ax = plt.subplots()
    ax.imshow(combined_image)
    ax.axis('off')

    # Add labels
    labels = {
        'separate': (images['separate'].width // 2, height - images['att_sep'].height),
        'att_allin': (images['separate'].width + images['att_allin'].width // 2, 
                      height - images['att_allin'].height),
        'att_sep': (images['att_sep'].width // 2, height),
        'noatt': (images['att_sep'].width + images['noatt'].width // 2, height),
        'allin': (images['allin'].width // 2, height)
    }
    
    for label_text, (x, y) in labels.items():
        ax.text(x, y + 10, label_text, ha='center')

    # Save
    output_path = OUT_SEPARATE_COMBINED / f"{id}.png"
    plt.savefig(str(output_path), bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(image_path(90, 'flicker8k'))