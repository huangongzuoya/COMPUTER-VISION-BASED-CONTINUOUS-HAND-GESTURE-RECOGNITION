import os
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from LSTM.util import DEVICE


def generate_increasing_sequence(seq_length, feature_num):
    # features = [np.arange(seq_length) + np.random.rand(seq_length) for _ in range(feature_num)]
    features = [np.arange(seq_length) + np.random.uniform(20, 50, size=seq_length) for _ in range(feature_num)]
    return np.column_stack(features)


def generate_decreasing_sequence(seq_length, feature_num):
    # features = [np.flip(np.arange(seq_length) + np.random.rand(seq_length)) for _ in range(feature_num)]
    features = [np.flip(np.arange(seq_length) + np.random.uniform(20, 50, size=seq_length)) for _ in range(feature_num)]
    # print(features)
    return np.column_stack(features)


def get_data_loader(num_samples, feature_num, seq_length, batch_size: int = 10) -> DataLoader[tuple[Tensor, ...]]:
    X_train = []
    y_train = []
    for _ in range(num_samples):
        if np.random.rand() > 0.5:
            sequence = generate_increasing_sequence(seq_length, feature_num)
            label = 0
        else:
            sequence = generate_decreasing_sequence(seq_length, feature_num)
            label = 1
        X_train.append(sequence)
        y_train.append(label)
    feature, label = np.array(X_train).astype(np.float32), np.array(y_train).astype(np.int64)
    device = DEVICE
    dataset = TensorDataset(torch.from_numpy(feature).to(device), torch.from_numpy(label).to(device))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == '__main__':
    num_samples = 100
    feature_num = 2
    seq_length = 20

    X_train = get_data_loader(num_samples, feature_num, seq_length)

    print(X_train)
    #
    # print(X_train[:3])
    # print(y_train[:3])

    # print("部分数据和标签：")
    # for i in range(5):
    #     print(f"数据 {i + 1}: {X_train[i]}，标签: {y_train[i]}")
