import json
import os
import warnings

import torch

from LSTM.DataGenerator.get_real_data import get_lstm_input_data
from LSTM.lstm import LSTMModel
from LSTM.util import DEVICE
from YOLOv5.yolov5 import run_yolov5_detection


class UnionPrediction:

    def __init__(self, yolov5_file_path: str, lstm_file_path: str, input_path: str, output_path: str, label_map_file_path: str = None) -> None:
        self.yolov5_path = yolov5_file_path
        self.lstm_file_path = lstm_file_path
        self.input_path = input_path
        self.output_path = output_path
        self.input_files = self.isValid()
        self.label_map = {}
        if label_map_file_path:
            with open(label_map_file_path, 'r') as label_map:
                self.label_map = json.load(label_map)

    def isValid(self) -> list[str]:
        # Verify YOLOv5 file
        if not os.path.exists(yolov5_file_path):
            raise FileNotFoundError(f"YOLOv5 file '{yolov5_file_path}' not found.")

        # Verify LSTM file
        if not os.path.exists(lstm_file_path):
            raise FileNotFoundError(f"LSTM file '{lstm_file_path}' not found.")

        # Verify input path
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path '{input_path}' not found.")

        # List files in input directory
        files = os.listdir(input_path)
        if not files:
            raise FileNotFoundError(f"No files found in '{input_path}")

        # Check if output directory exists, create if not
        if not os.path.exists(output_path):
            warnings.warn(f"Output directory '{output_path}' not found. Creating...")
            os.makedirs(output_path)

        # Process each file in the input directory
        return [os.path.join(input_path, file) for file in files]

    def predict(self, is_pred_one_out: bool = True) -> None:
        result = {}
        for file in self.input_files:
            filename = os.path.basename(file).split(".")[0]
            # detect gesture by yolov5
            run_yolov5_detection(file, self.yolov5_path, os.path.join(self.output_path, filename))
            yolov5_pred = os.path.join(self.output_path, filename, "labels")
            lstm_model = torch.load(self.lstm_file_path)
            lstm_model.eval()
            lstm_model.to(DEVICE)
            if is_pred_one_out:
                pred = self.lstm_pred_one_out(lstm_model, yolov5_pred)
                result[filename] = pred
            else:
                self.lstm_pred_step_by_step(lstm_model, yolov5_pred)

        with open(os.path.join(self.output_path, "Summary_of_Prediction.txt"), "w") as json_file:
            json.dump(result, json_file, indent=4)
        return

    def lstm_pred_one_out(self, lstm_model: LSTMModel, yolov5_out_folder: str) -> str:
        dataloader = get_lstm_input_data({
            yolov5_out_folder: 0
        }, batch_size=1)
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs, _ = lstm_model(inputs)
                _, predicted = torch.max(outputs, 1)
        print(f"predicted classes: {self.label_map.get(str(predicted.item()), predicted.item())}")
        return self.label_map.get(str(predicted.item()), predicted.item())

    def lstm_pred_step_by_step(self, lstm_model: LSTMModel, yolov5_out_folder: str) -> None:
        dataloader = get_lstm_input_data({
            yolov5_out_folder: 0
        })
        predictions = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                ht, ct = None, None
                for timestep in range(inputs.size(1)):  # Iterate over each timestep
                    input_timestep = inputs[:, timestep:(timestep + 1), :]
                    if ht is None and ct is None:
                        outputs, (ht, ct) = lstm_model(input_timestep)
                    else:
                        outputs, (ht, ct) = lstm_model(input_timestep, (ht, ct))
                    _, predicted = torch.max(outputs, 1)
                    predictions.append(self.label_map.get(str(predicted.item()), predicted.item()))
        print(f"predictions: {predictions}")
        return


if __name__ == '__main__':
    # define yolov5 model file path
    yolov5_file_path = "../YOLOv5_Detection/yolov5/runs/train/exp/weights/best.pt"
    # define lstm model file path
    lstm_file_path = "../LSTM/Output/05-11/LSTM_RealData_best.pt"
    # input folder (to be predicted)
    input_path = "DATA/INPUT"
    # output folder (save result)
    output_path = "DATA/OUTPUT"
    # label mapper file
    map_file = "../LSTM/train_label.json"
    up = UnionPrediction(yolov5_file_path, lstm_file_path, input_path, output_path, map_file)
    up.predict(is_pred_one_out=False)
