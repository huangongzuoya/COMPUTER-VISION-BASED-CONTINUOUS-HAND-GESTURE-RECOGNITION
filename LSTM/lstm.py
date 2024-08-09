import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from LSTM.DataGenerator.get_ecg_data import get_data_loader
from LSTM.DataGenerator.get_real_data import get_lstm_input_data
from LSTM.util import get_directory, DEVICE, plot_accuracy, plot_confusion_matrix, get_model_path


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x, init_states=None):
        # initialise
        if init_states is None:
            ht = torch.zeros(1, x.size(0), self.hidden_size).to(DEVICE)
            ct = torch.zeros(1, x.size(0), self.hidden_size).to(DEVICE)
        else:
            ht, ct = init_states

        # forward
        out, (ht_, ct_) = self.lstm(x, (ht, ct))
        out = self.fc(out[:, -1, :])  # Take the output of the last time point in the sequence as the prediction

        return out, (ht_, ct_)


def train_model(model, train_loader, val_loader, test_loader, name_list: list[str], num_epochs: int = 150,
                learning_rate: float = 0.001,
                early_stop_epochs: int = 30, train_name: str = "test"):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # cross entropy loss for integer labels
    # BCE Loss for one-hot encoding [with forward function output of： F.softmax(out, dim=1)]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    no_improvement_count = 0

    train_accuracies = []  # List to store training accuracies
    val_accuracies = []  # List to store validation accuracies

    save_path = get_directory()

    best_model = os.path.join(save_path, train_name + '_best.pt')
    last_model = os.path.join(save_path, train_name + '_last.pt')

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = correct / total

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        val_accuracy = evaluate_model(model, val_loader)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {accuracy:.4f} | Validation Accuracy: {val_accuracy:.4f}')
        train_accuracies.append(accuracy)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improvement_count = 0
            torch.save(model, best_model)  # Save the best model
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stop_epochs:
            print(f'Early stopping at epoch {epoch + 1} due to no improvement on validation accuracy. Best Accuracy: {best_accuracy:.4f}')
            break
        if 1 - accuracy < 10e-10 and 1 - val_accuracy < 10e-10:
            print(f'Early stopping at epoch {epoch + 1} due to both training and validation accuracy have reached their highest values. '
                  f'Best Accuracy: {best_accuracy:.4f}')
            break

    torch.save(model, last_model)  # Save the final model
    plot_accuracy({"Training Dataset": train_accuracies, "Validation Dataset": val_accuracies})
    test_model(best_model, model.output_size, test_loader, name_list)
    return


def evaluate_model(model, dataloader) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return accuracy


def test_model(model_path: str, output_size: int, dataloader, name_list: list[str]) -> None:
    target_model = torch.load(model_path)
    target_model.eval()

    confusion_matrix = np.zeros((output_size, output_size), dtype=float)
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs, _ = target_model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(targets)):
                confusion_matrix[predicted[i], targets[i]] += 1

            y_true += targets.tolist()
            y_pred += predicted.tolist()

    print(confusion_matrix)
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")
    # column sum
    col_sum = confusion_matrix.sum(axis=0)
    # Iterate through each element and divide by the sum of its column
    for i in range(output_size):
        confusion_matrix[:, i] /= col_sum[i]

    # Plot the confusion matrix
    plot_confusion_matrix(confusion_matrix, name_list)
    return


def test_model_in_each_data_point(dataloader) -> None:
    model_paths = get_model_path("_best.pt")
    if model_paths:
        model_path = model_paths[0]
    else:
        raise Exception("No models found")

    target_model = torch.load(model_path)
    target_model.eval()

    with torch.no_grad():
        for inputs, targets in dataloader:
            interval = int(inputs.size(1) * 0.05)
            correct = 0
            count = 0
            gather = []
            for timestep in range(inputs.size(1)):  # Iterate over each timestep
                input_timestep = inputs[:, timestep:(timestep + 1), :]
                outputs, _ = target_model(input_timestep)
                _, predicted = torch.max(outputs, 1)
                # print(f'timestep: {timestep}, predicted: {predicted}, target: {targets}')
                if predicted.item() == targets.item():
                    correct += 1

                count += 1
                if count % interval == 0:
                    gather.append(round(correct / interval, 4))
                    correct = 0
            print(f"Accuracy for {targets} sample: {gather}")


if __name__ == '__main__':
    # # initialize dataloader
    # input_size = 2
    # output_size = 5
    # seq_length = 4000
    # batch_size = 10
    #
    # # labels
    # name_list = ["ECG DATA"] + ["Mock DATA " + str(i) for i in range(1, output_size)]
    #
    # train_loader = get_data_loader(name_list, num_samples=6000, feature_num=input_size, seq_length=seq_length,
    #                                batch_size=batch_size, data_class=output_size)
    #
    # val_loader = get_data_loader(name_list, num_samples=2000, feature_num=input_size, seq_length=seq_length,
    #                              batch_size=batch_size, data_class=output_size, tag="Validation Dataset")
    #
    # test_loader = get_data_loader(name_list, num_samples=1000, feature_num=input_size, seq_length=seq_length,
    #                               batch_size=batch_size, data_class=output_size, tag="Test Dataset")
    #
    # # train model
    # # # initialize model
    # model = LSTMModel(input_size, 64, output_size)
    # # higher hidden size refer to more info to store in the memory
    #
    # enable_training = True
    #
    # if enable_training:
    #     train_model(model, train_loader, val_loader, test_loader, name_list, train_name="LSTM_Validation")
    #
    # # evaluate model
    #
    # test_predication_dataloader = get_data_loader(name_list, num_samples=100, feature_num=input_size,
    #                                               seq_length=seq_length, batch_size=1, data_class=output_size,
    #                                               tag="test predication dataset")
    # test_model_in_each_data_point(test_predication_dataloader)

    # real data test
    input_size = 6
    output_size = 2

    with open('train_info.json', 'r') as train_info:
        train_desc = json.load(train_info)
    with open('val_info.json', 'r') as val_info:
        val_desc = json.load(val_info)
    with open('test_info.json', 'r') as test_info:
        test_desc = json.load(test_info)
    with open('train_label.json', 'r') as data_label:
        action_label = json.load(data_label)
    train_loader = get_lstm_input_data(train_desc)
    val_loader = get_lstm_input_data(val_desc)
    test_loader = get_lstm_input_data(test_desc)

    model = LSTMModel(input_size, 32, output_size)
    train_model(model, train_loader, val_loader, test_loader, list(action_label.values()), train_name="LSTM_RealData")

    # test_model_in_each_data_point(get_lstm_input_data(
    #     {
    #         "D:\\PycharmProjects\\Semester 3\\GIT\\Capstone\\YOLOv5\\DATA\\OUTPUT\\video51\\exp\\best\\labels": 0,
    #         "D:\\PycharmProjects\\Semester 3\\GIT\\Capstone\\YOLOv5\\DATA\\OUTPUT\\video52\\exp\\best\\labels": 1,
    #     },
    #     batch_size=1
    # ))
