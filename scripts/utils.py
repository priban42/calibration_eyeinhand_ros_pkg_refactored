import cv2
import yaml
import datetime
from pathlib import Path
from collections import OrderedDict


def get_images_paths(data_dir: Path):
    return sorted([path for path in data_dir.iterdir() if path.is_file() and path.suffix in {'.jpg', '.jpeg', '.png'}])


def load_images(data_dir, cropout_hand = True):
    image_paths = get_images_paths(data_dir)
    img_file_map = OrderedDict()
    for img_path in image_paths:
        img_name = img_path.stem
        img_file_map[img_name] = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        if cropout_hand:
            img = img_file_map[img_name]
            height = img.shape[0]
            width = img.shape[1]
            # print("width:", type(width), width)
            image = cv2.rectangle(img_file_map[img_name], (width//2 - 250, 0), (width//2 + 150, 120), (100, 100, 100), -1)
            img_file_map[img_name] = image

    return img_file_map

def save_yaml(file_path, data):
    with open(file_path, 'w') as f:
        yaml.dump(data, f)


def read_yaml(file_path):
    with open(file_path,'r') as f:
        return yaml.load(f,yaml.SafeLoader)


def get_time_str(form):
    return str(datetime.datetime.now().strftime(form))
