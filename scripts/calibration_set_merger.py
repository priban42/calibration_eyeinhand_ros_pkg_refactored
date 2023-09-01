import os
import shutil
import random
import yaml
from pathlib import Path

def get_all_entries(base_path, set_names):
    entries = []
    for set_name in set_names:
        path = os.path.join(base_path, set_name)
        files = os.listdir(path)
        for file in files:
            if file.endswith(".jpg"):
                entries.append(path + "/" + file.split('.')[0])
    return entries

def copy_edited_yaml(input_path, output_path, entry_to_edit):
    with open(input_path, "r") as stream:
        imported_yaml = yaml.safe_load(stream)
    for key in entry_to_edit.keys():
        imported_yaml[key] = entry_to_edit[key]
    with open(output_path, "w") as output:
        yaml.safe_dump(imported_yaml, output, default_style=None)


def copy_entries_to_dir(entries, dir_path):
    for entry_count, entry in enumerate(entries):
        new_name = f'{entry_count:04}'
        shutil.copy2(entry + ".jpg", dir_path +"/"+ new_name + ".jpg")
        copy_edited_yaml(entry + ".yaml", dir_path +"/"+ new_name + ".yaml", {"image_name": new_name, "directory":dir_path})

def get_chunk(arr, i, chunks, chunk_sizes = None):
    if chunk_sizes != None:
        return arr[int(len(arr)*sum(chunk_sizes[:i])):int(len(arr)*sum(chunk_sizes[:i+1]))]
    if i == chunks-1:
        return arr[len(arr)//chunks*i:len(arr)]
    return arr[len(arr)//chunks*i:len(arr)//chunks*(i+1)]

def refresh_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def main():
    # BASE_PATH = "/home/bagr/ws_moveit/src/calibration_eyeinhand_ros_pkg/calibration"
    BASE_PATH = "C:\\Users\\Vojta\\PycharmProjects\\calibration_eyeinhand_ros_pkg\\calibration"
    SPLIT = 1/3
    # new_set_names = ["calibration_merged_1", "calibration_merged_2"]
    new_set_names = ["calibration_d455_merged"]
    sets_to_merge = ["calibration_d455_high_res_1", "calibration_d455_high_res_2"]
    entries = get_all_entries(BASE_PATH, sets_to_merge)
    random.shuffle(entries)
    for name in new_set_names:
        refresh_path(os.path.join(BASE_PATH, name))

    for i, new_set_name in enumerate(new_set_names):
        chunks = len(new_set_names)
        split_entry = get_chunk(entries[0:int(len(entries)*SPLIT)], i, chunks)
        new_set_path = os.path.join(BASE_PATH, new_set_name)
        copy_entries_to_dir(split_entry, new_set_path)

def random_split(seed):
    # BASE_PATH = "/home/bagr/ws_moveit/src/calibration_eyeinhand_ros_pkg/calibration"
    BASE_PATH = "C:\\Users\\Vojta\\PycharmProjects\\calibration_eyeinhand_ros_pkg\\calibration"
    SPLIT = 1
    # CHUNK_SIZES = [0.6, 0.4] # len = len(new_set_names), sum = 1
    CHUNK_SIZES = [1]
    new_set_names = ["calibration_merged"]
    # new_set_names = ["calibration_d455_train", "calibration_d455_test"]
    sets_to_merge = ["calibration_d455_high_res_1", "calibration_d455_high_res_2"]
    entries = get_all_entries(BASE_PATH, sets_to_merge)
    random.seed(seed)
    random.shuffle(entries)
    for name in new_set_names:
        refresh_path(os.path.join(BASE_PATH, name))

    for i, new_set_name in enumerate(new_set_names):
        chunks = len(new_set_names)
        split_entry = get_chunk(entries[0:int(len(entries)*SPLIT)], i, chunks, CHUNK_SIZES)
        new_set_path = os.path.join(BASE_PATH, new_set_name)
        copy_entries_to_dir(split_entry, new_set_path)

if __name__ == '__main__':
    random_split(0)
    # main()