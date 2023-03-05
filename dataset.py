import os
from typing import Tuple

import chess
import pandas
import torch
from pandas import read_csv, DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from utils.to_tensor import fen_to_tensor, result_to_tensor, create_action_mask, time_to_tensor
from utils.flip import flip_uci
from utils.uci_to_action import uci_to_action

COLUMNS = ["Position", "Action mask", "Time", "Move", "Result", "Percentage of time used"]
COLUMNS_TIME = [1, 2, 3]


class DatasetMimic(Dataset):
    """
    Class that represents the dataset to use for the training
    """

    def __init__(self, csv_folder: str):
        """
        Creates a dataset from a folder with csv files.
        The csv files must contain the following columns in the following order:

        - FEN: FEN of the board
        - Time remaining self: time remaining for the player to move
        - Time remaining opponent: time remaining for the opponent to move
        - Increment: time increment for both players
        - Move: Move to take (in UCI format)
        - Result: An integer representing the result of the game (0: loss, 1: draw, 2: win)
        - Time used: Time used by the player to make the move

        :param csv_folder: folder that contains the csv files
        """
        self.data = DataFrame(columns=COLUMNS)

        for file in os.listdir(csv_folder):

            if not file.endswith(".csv"):
                continue

            csv_data = read_csv(f"{csv_folder}/{file}", header=None)
            self.data = pandas.concat([self.data, transform_data(csv_data)], ignore_index=True)
            print(f"Loaded {len(csv_data)} positions from {file}")

    def __len__(self) -> int:
        """
        Overrides the __len__ method of the Dataset class

        :return: the number of elements in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Overwrites the __getitem__ method of the Dataset class

        :param idx: index of the element to get
        :return: a tuple with the input and the output of the neural network
        """
        position = self.data.iloc[idx]["Position"]
        action_mask = self.data.iloc[idx]["Action mask"]
        times = self.data.iloc[idx]["Time"]

        output_move = self.data.iloc[idx]["Move"]
        output_result = self.data.iloc[idx]["Result"]
        output_time = self.data.iloc[idx]["Percentage of time used"]

        return (position, action_mask, times), (output_move, output_result, output_time)


def transform_data(csv_data):
    data = DataFrame(columns=COLUMNS)

    data["Position"] = csv_data[0].transform(fen_to_tensor)
    data["Action mask"] = csv_data[0].transform(create_action_mask)
    data["Time"] = csv_data[COLUMNS_TIME].values.tolist()
    data["Time"] = data["Time"].transform(time_to_tensor)

    csv_data[7] = csv_data[0].transform(lambda x: x.split(" ")[1] == "b")
    csv_data.loc[csv_data[7], 4] = csv_data.loc[csv_data[7], 4].transform(flip_uci)
    data["Move"] = csv_data[4].transform(uci_to_action)
    data["Result"] = csv_data[5].transform(result_to_tensor)
    data["Percentage of time used"] = csv_data[6] / (csv_data[1] + 1e-6)  # + 1e-6 to avoid division by 0
    data["Percentage of time used"] = data["Percentage of time used"].transform(
        lambda x: torch.tensor(x, dtype=torch.float32)
    )

    return data
