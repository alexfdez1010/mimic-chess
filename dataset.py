import os
from typing import Tuple

import chess
import pandas
import torch
from pandas import read_csv, DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from utils.constants import NUM_ACTIONS
from utils.fen_to_tensor import fen_to_tensor
from utils.flip import flip_uci
from utils.uci_to_action import uci_to_action

COLUMNS = ["Position", "Action mask", "Time", "Move", "Result", "Percentage of time used"]
COLUMNS_TIME = [1, 2, 3]

MAX_TIME = 7200
MAX_INCREMENT = 60


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
        Sobreescribe el método __len__ de la clase Dataset

        :return: el número de elementos del dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Sobreescribe el método __getitem__ de la clase Dataset

        :param idx: índice del elemento a obtener
        :return: el elemento del dataset, siempre desde la perspectiva de las blancas
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


def result_to_tensor(result: int) -> Tensor:
    """
    Transform a result into a tensor

    :param result: result
    :return: tensor with the result
    """
    return torch.tensor(result, dtype=torch.long)


def create_action_mask(fen: str) -> Tensor:
    """
    Creates an action mask from a FEN

    :param fen: FEN of the board
    :return: tensor with the action mask
    """
    action_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    is_white = fen.split(" ")[1] == "w"

    board = chess.Board(fen)
    func = lambda x: uci_to_action(x.uci()) if is_white else uci_to_action(flip_uci(x.uci()))

    legal_actions = list(map(func, board.legal_moves))

    action_mask[legal_actions] = 1

    return action_mask


def time_to_tensor(times: list) -> Tensor:
    """
    Transforms a list of times into a tensor

    :param times: list of times
    :return: tensor with the times
    """
    times[0] = times[0] / MAX_TIME
    times[1] = times[1] / MAX_TIME
    times[2] = times[2] / MAX_INCREMENT

    return torch.tensor(times, dtype=torch.float32)


def transform_flip_uci(row: pandas.Series) -> pandas.Series:
    """
    Flip a UCI if the position is from the black player

    :param row: row of the dataset
    :return: row of the dataset with the UCI flipped if the position is from the black player
    """
    is_white = row[0].split(" ")[1] == "w"
    if not is_white:
        print(f"{row[4]} -> {flip_uci(row[4])}")

    row[4] = row[4] if is_white else flip_uci(row[4])
    print(row[4])

    return row
