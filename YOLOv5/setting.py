import os
import re
import sys

import torch
from torch import device

model_file = "../YOLOv5_Detection/yolov5/runs/train/exp/weights/best.pt"
default_root = os.path.join(os.getcwd(), "DATA")
default_input = os.path.join(default_root, "INPUT")
default_output = os.path.join(default_root, "OUTPUT")


def check_dir_structure() -> None:
    if not os.path.exists(default_root):
        os.makedirs(default_root)
    if not os.path.exists(default_input):
        os.makedirs(default_input)
    if not os.path.exists(default_output):
        os.makedirs(default_output)


def get_file_path(filename: str) -> str:
    file_path = os.path.join(default_input, filename)
    if os.path.exists(file_path):
        return file_path
    else:
        print(f"Error: File '{filename}' not found in '{default_input}'.")
        sys.exit(1)


def get_output_dir(video_filename: str, model_filename: str) -> str:
    video_basename, _ = os.path.splitext(video_filename)
    model_basename, _ = os.path.splitext(os.path.basename(model_filename))

    pattern = r'exp\d*'
    match = re.search(pattern, model_filename)
    if match:
        tag = match.group()
    else:
        tag = "default"

    output_dir = os.path.join(default_output, video_basename, tag, model_basename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


if __name__ == "__main__":
    # print(os.path.exists(model_file))
    # check_dir_structure()
    get_output_dir("video.mp4", model_file)

