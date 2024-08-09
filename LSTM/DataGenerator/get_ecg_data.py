import warnings
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
from enum import Enum, auto
from itertools import islice

from LSTM.util import DEVICE, plot_data_frequency


def generate_ecg_data(duration_sec: float, sampling_rate: int = 1000, noise_level: float = 0.01) -> (
        tuple)[np.ndarray, np.ndarray]:
    """
    !! NO LONGER NEED !!
    Args:
        duration_sec:
        sampling_rate:
        noise_level:

    Returns:
        t: a serial of time stamp
        ecg_with_noise: a serial of ecg data with noise
    """
    # Generate time axis
    t = np.linspace(0, duration_sec, int(duration_sec * sampling_rate), endpoint=False)

    # Generate ECG waveform
    ecg_waveform = (
            np.sin(2 * np.pi * 0.5 * t) +  # P wave
            np.sin(2 * np.pi * 1.0 * t) +  # QRS complex
            0.6 * np.sin(2 * np.pi * 1.5 * t) +  # T wave
            0.3 * np.sin(2 * np.pi * 2.0 * t)  # U wave
    )

    # Add noise
    noise = np.random.normal(0, noise_level, ecg_waveform.shape)
    ecg_with_noise = ecg_waveform + noise
    return t, ecg_with_noise


class WaveMode(Enum):
    ECG = 0
    SIN = 1
    SIN_COS = 2
    SQUARE = 3
    SAWTOOTH = 4


def generate_other_data(duration_sec: float, sampling_rate: int = 1000, noise_level: float = 0.01,
                        mode: WaveMode = WaveMode.SIN) -> tuple[np.ndarray, np.ndarray]:
    # Generate time axis
    t = np.linspace(0, duration_sec, int(duration_sec * sampling_rate), endpoint=False)

    if mode == WaveMode.ECG:
        other_waveform = (
            np.sin(2 * np.pi * 0.5 * t) +  # P wave
            np.sin(2 * np.pi * 1.0 * t) +  # QRS complex
            0.6 * np.sin(2 * np.pi * 1.5 * t) +  # T wave
            0.3 * np.sin(2 * np.pi * 2.0 * t)  # U wave
    )
    elif mode == WaveMode.SIN:
        other_waveform = (
            np.sin(2 * np.pi * 0.5 * t)
        )
    elif mode == WaveMode.SIN_COS:
        other_waveform = (
                np.sin(2 * np.pi * 0.5 * t) +  # Sinusoidal component
                np.cos(2 * np.pi * 0.3 * t)  # Random noise
        )
    elif mode == WaveMode.SQUARE:
        other_waveform = (
            np.sign(np.sin(2 * np.pi * 0.5 * t))
        )
    elif mode == WaveMode.SAWTOOTH:
        other_waveform = (
            np.abs(signal.sawtooth(2 * np.pi * 0.5 * t))
        )
    else:
        raise ValueError(f"Unknown mode: {mode}, use mode = 1")

    # Add noise
    noise = np.random.normal(0, noise_level, other_waveform.shape)
    waveform_with_noise = other_waveform + noise
    return t, waveform_with_noise


def extract_random_segment(array, sample_length: int = 2500):
    """
    Random segmentation of the original array with the length of sample length
    Args:
        array:
        sample_length:

    Returns:
        array:
    """
    array_length = len(array)
    if array_length < sample_length:
        raise ValueError(f"Array length must be greater or equal than {sample_length}.")

    # Generate a random starting index within the valid range
    start_index = np.random.randint(0, array_length - sample_length)

    # Extract a segment of length 1000 starting from the random index
    random_segment = array[start_index:(start_index + sample_length)]

    return random_segment


def get_data_loader(name_list: list[str], num_samples: int, feature_num: int, seq_length: int = 6000,
                    batch_size: int = 10, data_class: int = 2, tag: str = "Training Dataset") -> DataLoader[
    tuple[Tensor, ...]]:
    """

    Args:
        name_list: This is a list of names of different classes
        num_samples:
        feature_num: This is the dimension of feature vectors
        seq_length:
        batch_size:
        data_class: does that means the number of class?
        tag:

    Returns:

    """
    # _, ecg_data = generate_ecg_data(seq_length / 1000)
    other_date_list = []
    label_list = []
    for mode in islice(WaveMode, data_class):
        _, other_data = generate_other_data(seq_length / 1000, mode=mode)
        other_date_list.append(other_data)
        label_list.append(mode.value)
    x_train = []
    y_train = []
    for _ in range(num_samples):
        random_number = random.randint(0, data_class - 1)       # why should data_class - 1 ?
        # if random_number == 0:
        #     sequence = np.column_stack([extract_random_segment(ecg_data) for _ in range(feature_num)])
        #     label = 0
        # else:
        #     sequence = np.column_stack([extract_random_segment(other_date_list[random_number - 1])
        #                                 for _ in range(feature_num)])
        #     label = random_number
        sequence = np.column_stack([extract_random_segment(other_date_list[random_number]) for _ in range(feature_num)])
        label = label_list[random_number]

        x_train.append(sequence)
        y_train.append(label)

    feature, label = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.int64)
    device = DEVICE
    dataset = TensorDataset(torch.from_numpy(feature).to(device), torch.from_numpy(label).to(device))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    plot_data_frequency(y_train, name_list, tag=tag)
    return data_loader


if __name__ == '__main__':
    get_data_loader("some_names",num_samples=6000, feature_num=2, seq_length=3000, batch_size=10,
                    data_class=5)
    get_data_loader("some_names",num_samples=1000, feature_num=2, seq_length=3000, batch_size=10,
                    data_class=5, tag="ValidationDataset")
