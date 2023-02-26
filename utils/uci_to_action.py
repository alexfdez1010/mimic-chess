from typing import Tuple, Optional

import chess
import torch

from utils.constants import ROWS, COLS, NUM_MOVES, NUM_ACTIONS

KNIGHT_MOVES = {
    (2, 1): 0,
    (1, 2): 1,
    (-1, 2): 2,
    (-2, 1): 3,
    (-2, -1): 4,
    (-1, -2): 5,
    (1, -2): 6,
    (2, -1): 7
}

PROMOTIONS = {
    'n': 0,
    'b': 1,
    'r': 2,
}

DIRECTIONS = [
    (1, 0),  # N
    (1, 1),  # NE
    (0, 1),  # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1),  # SO
    (0, -1),  # O
    (1, -1)  # NO
]

KNIGHT_MOVES_INVERSE = {
    0: (2, 1),
    1: (1, 2),
    2: (-1, 2),
    3: (-2, 1),
    4: (-2, -1),
    5: (-1, -2),
    6: (1, -2),
    7: (2, -1)
}


def valid_square(row: int, col: int) -> bool:
    """
    Check whether a square is valid
    :param row: row of the square
    :param col: column of the square
    :return: true if the square is valid, false otherwise
    """
    return 0 <= row < ROWS and 0 <= col < COLS


def create_maps() -> Tuple[dict, dict]:
    """
    Creates the maps to convert from tuple to action and viceversa
    :return: tuple to action map and action to tuple map
    """
    tuple_to_action_map = {}
    action_to_tuple_map = {}
    counter = 0

    for move in range(NUM_MOVES):

        if move < 56:
            x = DIRECTIONS[move // 7][0] * ((move % 7) + 1)
            y = DIRECTIONS[move // 7][1] * ((move % 7) + 1)
        elif move < 64:
            x = KNIGHT_MOVES_INVERSE[move - 56][0]
            y = KNIGHT_MOVES_INVERSE[move - 56][1]
        else:
            x = 1
            y = ((move - 64) % 3) - 1

        for row in range(ROWS):

            if move >= 64 and row != 6:
                continue

            for col in range(COLS):

                new_row = row + x
                new_col = col + y

                if valid_square(new_row, new_col):
                    tuple_to_action_map[(move, row, col)] = counter
                    action_to_tuple_map[counter] = (move, row, col)
                    counter += 1

    return tuple_to_action_map, action_to_tuple_map


TUPLE_TO_ACTION_MAP, ACTION_TO_TUPLE_MAP = create_maps()


def uci_to_tuple(uci: str) -> Tuple[int, int, int]:
    """
    Converts a UCI string to a tuple
    :param uci: UCI string
    :return: tuple with the move, row and column in that order
    """
    row_from = ord(uci[1]) - ord('1')
    col_from = ord(uci[0]) - ord('a')

    row_to = ord(uci[3]) - ord('1')
    col_to = ord(uci[2]) - ord('a')

    move: int

    if len(uci) == 5 and uci[4] != 'q':
        move = 64 + PROMOTIONS[uci[4]] * 3 + 1 + (col_to - col_from)

    elif col_from == col_to:
        if row_to > row_from:
            move = row_to - row_from - 1
        else:
            move = 7 * 4 + row_from - row_to - 1

    elif row_from == row_to:
        if col_to > col_from:
            move = 7 * 2 + col_to - col_from - 1
        else:
            move = 7 * 6 + col_from - col_to - 1

    elif row_to - row_from == col_to - col_from:
        if row_to > row_from:
            move = 7 * 1 + row_to - row_from - 1
        else:
            move = 7 * 5 + row_from - row_to - 1

    elif COLS - (1 + row_from + col_from) == COLS - (1 + row_to + col_to):
        if col_to > col_from:
            move = 7 * 3 + col_to - col_from - 1
        else:
            move = 7 * 7 + col_from - col_to - 1

    else:
        move = 7 * 8 + KNIGHT_MOVES[(row_to - row_from, col_to - col_from)]

    return move, row_from, col_from


def tuple_to_action(tuple_move: Tuple[int, int, int]) -> int:
    """
    Converts a tuple to the corresponding action of the neural network
    :param tuple_move: tuple with the move, row and column in that order
    :return: the corresponding action of the neural network
    """
    return TUPLE_TO_ACTION_MAP.get(tuple_move, -1)


def action_to_tuple(action: int) -> Tuple[int, int, int]:
    """
    Converts an action of the neural network to the corresponding tuple
    :param action: action of the neural network
    :return: tuple with the move, row and column in that order
    """
    return ACTION_TO_TUPLE_MAP.get(action, (-1, -1, -1))


def uci_to_action(uci: str) -> int:
    """
    Converts a UCI string to the corresponding action of the neural network
    :param uci: UCI string
    :return: the corresponding action of the neural network
    """
    return tuple_to_action(uci_to_tuple(uci))


def action_to_uci(action: int) -> str:
    """
    Converts an action of the neural network to the corresponding UCI string
    :param action: action of the neural network
    :return: UCI string
    """
    return tuple_to_uci(action_to_tuple(action))


def tuple_to_uci(tuple_move: Tuple[int, int, int]) -> Optional[str]:
    """
    Converts a tuple to the corresponding UCI string
    :param tuple_move: tuple with the move, row and column in that order
    :return: UCI string
    """
    if tuple == (-1, -1, -1):
        return None

    move, row_from, col_from = tuple_move

    row_from_uci = chr(ord('1') + row_from)
    col_from_uci = chr(ord('a') + col_from)

    promotion = ''

    if move < 56:
        row_to = row_from + DIRECTIONS[move // 7][0] * ((move % 7) + 1)
        col_to = col_from + DIRECTIONS[move // 7][1] * ((move % 7) + 1)
    elif move < 64:
        row_to = row_from + KNIGHT_MOVES_INVERSE[move - 56][0]
        col_to = col_from + KNIGHT_MOVES_INVERSE[move - 56][1]
    else:
        row_to = row_from + 1
        col_to = ((move - 64) % 3) - 1 + col_from
        promotion = (move - 64) // 3

        for p in PROMOTIONS:
            if promotion == PROMOTIONS[p]:
                promotion = p
                break

    row_to_uci = chr(ord('1') + row_to)
    col_to_uci = chr(ord('a') + col_to)

    return f'{col_from_uci}{row_from_uci}{col_to_uci}{row_to_uci}{promotion}'


def create_policy_matrix() -> torch.Tensor:
    """
    Creates a matrix that converts a policy with 4672 actions to a policy with 1858 actions
    :return: policy matrix
    """
    policy_matrix = torch.zeros((NUM_MOVES * ROWS * COLS, NUM_ACTIONS), dtype=torch.float32)

    for (move, row, col), action in TUPLE_TO_ACTION_MAP.items():
        policy_matrix[move * ROWS * COLS + row * COLS + col][action] = 1

    return policy_matrix


def add_promotion(uci_move: str, board: chess.Board) -> str:
    """
    Adds a promotion to a UCI move if it is necessary
    :param uci_move: Move in UCI format
    :param board: Board where the move is played
    :return: Move in UCI format with promotion
    """
    square = chess.parse_square(uci_move[0:2])
    piece = board.piece_at(square)

    if (
        piece and piece.piece_type == chess.PAWN
            and len(uci_move) == 4
            and ((uci_move[1] == '7' and uci_move[3] == '8') or (uci_move[1] == '2' and uci_move[3] == '1'))
    ):
        return uci_move + "q"

    return uci_move
