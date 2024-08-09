import glob
import json
import os
import warnings
from typing import List, Union, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from LSTM.util import DEVICE


def transform_single_yolov5_video_output(txt_file_path: str, valid_class: List[str] = None,
                                         box_num: int = 2) -> Union[list[list[float]], tuple[list[list[float]], Any]]:
    x_train = []
    txt_files = glob.glob(txt_file_path + "/*.txt")
    if not txt_files:
        warnings.warn(f'{txt_file_path} has no txt files')
        return x_train

    # get video filename and total_frames
    prefix, _ = os.path.basename(txt_files[-1]).split(".")[0].split("_")
    for file_path in txt_files:
        with open(file_path, 'r') as file:
            content = file.read()
            content = content.rstrip()

            contents = content.split('\n')
            valid_contents = []
            # only accept label in valid_class
            if valid_class:
                for content in contents:
                    if content[0] in valid_class:
                        valid_contents.append(content)
                contents = valid_contents
            # do not consider blank frame
            # we only select one box/gesture
            if len(contents) > 0:
                x_train.append([float(item) for item in contents[0].split(" ")])

    return x_train


def get_lstm_input_data(data_desc: dict[str, int], batch_size: int = 10) -> DataLoader[tuple[Tensor, ...]]:
    """
    Args:
        data_desc: map video file path to action label and frame number
    eg: {'DATA/OUTPUT/video1/exp/best/labels': 1}
    Returns:
        Dataloader
    """
    x_train, y_train = [], []
    for video_file_path, label in data_desc.items():
        sub_train = transform_single_yolov5_video_output(video_file_path, valid_class=['3'])
        # filter out samples with length less than 40 data points
        if len(sub_train) >= 30:
            x_train.append(sub_train)
            y_train.append(label)

    len_of_video = [len(i) for i in x_train]

    # print(f"len of each video: {len_of_video}")
    # data_series = pd.Series(len_of_video)
    # stats = data_series.describe()
    # print(stats)

    max_len = max(len_of_video)
    x_train_interpolated = []
    for x_i in x_train:
        if len(x_i) < max_len:
            x_i = linear_interpolation(x_i, target_num=max_len)
        x_train_interpolated.append(x_i)

    feature, label = np.array(x_train_interpolated).astype(np.float32), np.array(y_train).astype(np.int64)
    dataset = TensorDataset(torch.from_numpy(feature).to(DEVICE), torch.from_numpy(label).to(DEVICE))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def linear_interpolation(data_list: List[List[Union[int, float]]], target_num: int) -> List[List[Union[int, float]]]:
    if len(data_list) < 2:
        raise ValueError('data_list must have at least two elements')

    while target_num >= 2 * len(data_list) - 1:
        tmp_list = []
        for index in range(len(data_list) - 1):
            tmp_list.append(data_list[index])
            new_record = [(data_list[index][index2] + data_list[index + 1][index2])/2 for index2 in range(len(data_list[index]))]
            tmp_list.append(new_record)
        tmp_list.append(data_list[-1])
        data_list = tmp_list.copy()

    diff = target_num - len(data_list)
    result_list = []
    for index in range(len(data_list)):
        result_list.append(data_list[index])
        if index < diff:
            new_record = [(data_list[index][index2] + data_list[index + 1][index2])/2 for index2 in range(len(data_list[index]))]
            result_list.append(new_record)
    return result_list


if __name__ == "__main__":
    # print(transform_single_yolov5_video_output("../../YOLOv5/DATA/OUTPUT/video1/exp/best/labels", 1, 300))
    # print(get_lstm_input_data(
    #     {
    #         "D:\\PycharmProjects\\Semester 3\\GIT\\Capstone\\YOLOv5\\DATA\\OUTPUT\\video213\\exp\\best\\labels": 0,
    #         "D:\\PycharmProjects\\Semester 3\\GIT\\Capstone\\YOLOv5\\DATA\\OUTPUT\\video214\\exp\\best\\labels": 0,
    #
    #         "D:\\PycharmProjects\\Semester 3\\GIT\\Capstone\\YOLOv5\\DATA\\OUTPUT\\video216\\exp\\best\\labels": 0,
    #         "D:\\PycharmProjects\\Semester 3\\GIT\\Capstone\\YOLOv5\\DATA\\OUTPUT\\video217\\exp\\best\\labels": 0,
    #     }
    # ))

    with open('../train_info.json', 'r') as train_info:
        train_desc = json.load(train_info)
    print(get_lstm_input_data(
        train_desc
    ))

    # test_input = [
    #     [3, 0.1, 0.1, 0.1, 0.1, 0.8],
    #     [3, 0.2, 0.2, 0.2, 0.2, 0.8],
    #     [3, 0.3, 0.3, 0.3, 0.3, 0.8],
    #     [3, 0.4, 0.4, 0.4, 0.4, 0.8]
    # ]
    # test_output = linear_interpolation(test_input, target_num=10)
    # for t_o in test_output:
    #     print(f"test output: {t_o}")
