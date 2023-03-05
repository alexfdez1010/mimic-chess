import chess
import torch
from torch import zeros, Tensor

from utils.constants import ROWS, COLS, CHANNELS, NUM_ACTIONS, SECONDS_IN_HOUR, SECONDS_IN_MINUTE
from utils.flip import flip_fen, flip_uci
from utils.uci_to_action import uci_to_action

PIECE_TO_CHANNEL = {
    'P': 0,
    'N': 1,
    'B': 2,
    'R': 3,
    'Q': 4,
    'K': 5,
    'p': 6,
    'n': 7,
    'b': 8,
    'r': 9,
    'q': 10,
    'k': 11
}  # map between pieces and channels

CASTLE_TO_CHANNEL = {
    'K': 12,
    'Q': 13,
    'k': 14,
    'q': 15
}  # map between castles and channels

COLOUR = 16  # colour channel

PADDED_CONVOLUTION = 17  # padded convolution channel

MAX_TIME = 7200  # max time allowed for a move in seconds
MAX_INCREMENT = 60  # max increment allowed for a move in seconds


def fen_to_tensor(fen: str) -> Tensor:
    """
    Converts a FEN string to a tensor for the neural network
    :param fen: FEN string
    :return: tensor for the neural network
    """
    tensor = zeros(CHANNELS, ROWS, COLS, dtype=torch.float32)
    tensor[PADDED_CONVOLUTION] = 1

    if fen.split(' ')[1] == 'b':
        fen = flip_fen(fen)

    board, color, castles, en_passant, _, _ = fen.split(' ')

    tensor[COLOUR] = 1 if color == 'b' else 0

    i, j = ROWS - 1, 0

    for ch in board:
        if ch == '/':
            i -= 1
            j = 0
        elif ch.isalpha():
            tensor[PIECE_TO_CHANNEL[ch], i, j] = 1
            j += 1
        else:
            j += int(ch)

    if castles != '-':
        for castle in castles:
            tensor[CASTLE_TO_CHANNEL[castle]] = 1

    if en_passant != '-':
        if en_passant[1] == '6':
            tensor[PIECE_TO_CHANNEL['p'], 7, ord(en_passant[0]) - ord('a')] = 1
            tensor[PIECE_TO_CHANNEL['p'], 4, ord(en_passant[0]) - ord('a')] = 0
        else:
            tensor[PIECE_TO_CHANNEL['P'], 0, ord(en_passant[0]) - ord('a')] = 1
            tensor[PIECE_TO_CHANNEL['P'], 3, ord(en_passant[0]) - ord('a')] = 0

    return tensor


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


def time_string_to_seconds(time: str) -> int:
    """
    Converts a time in string format to seconds
    :param time: time in string format
    :return: time in seconds
    """
    hours, minutes, seconds = list(map(int, time.split(":")))
    return hours * SECONDS_IN_HOUR + minutes * SECONDS_IN_MINUTE + seconds


def seconds_to_time(seconds: int) -> str:
    """
    Converts seconds to a time in string format
    :param seconds: time in seconds
    :return: time in string format
    """
    hours = seconds // SECONDS_IN_HOUR
    minutes = (seconds % SECONDS_IN_HOUR) // SECONDS_IN_MINUTE
    seconds = seconds % SECONDS_IN_MINUTE

    return f"{hours}:{minutes}:{seconds}"
