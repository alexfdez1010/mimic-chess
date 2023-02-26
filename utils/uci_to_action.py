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
    Comprueba si una casilla es válida
    :param row: fila de la casilla
    :param col: columna de la casilla
    :return: True si la casilla es válida, False en caso contrario
    """
    return 0 <= row < ROWS and 0 <= col < COLS


def create_maps() -> Tuple[dict, dict]:
    """
    Crea los diccionarios para pasar de tupla a acción y de acción a tupla
    :return: dos diccionarios de conversion de UCI a tensor y viceversa
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
    Convierte una cadena en formato UCI en el tensor correspondiente para la red neuronal
    :param uci: una cadena en formato UCI
    :return: una tupla indicando el movimiento, la fila y la columna
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
    Convierte una tupla a la acción correspondiente de la red neuronal
    :param tuple_move: tupla a la que se ha convertido el movimiento uci
    :return: la acción correspondiente de la red neuronal
    """
    return TUPLE_TO_ACTION_MAP.get(tuple_move, -1)


def action_to_tuple(action: int) -> Tuple[int, int, int]:
    """
    Convierte una acción a la tupla correspondiente del movimiento uci
    :param action: acción de la red neuronal
    :return: la tupla correspondiente del movimiento uci
    """
    return ACTION_TO_TUPLE_MAP.get(action, (-1, -1, -1))


def uci_to_action(uci: str) -> int:
    """
    Convierte una cadena en formato UCI en la acción correspondiente de la red neuronal
    :param uci: una cadena en formato UCI
    :return: la acción correspondiente de la red neuronal
    """
    return tuple_to_action(uci_to_tuple(uci))


def action_to_uci(action: int) -> str:
    """
    Convierte una acción a la cadena en formato UCI correspondiente
    :param action: acción de la red neuronal
    :return: una cadena en formato UCI
    """
    return tuple_to_uci(action_to_tuple(action))


def tuple_to_uci(tuple_move: Tuple[int, int, int]) -> Optional[str]:
    """
    Convierte una tupla a la cadena correspondiente en formato UCI
    :param tuple_move: tupla a la que se ha convertido el movimiento uci
    :return: la cadena correspondiente en formato UCI
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
    Crea la matriz que permite convertir la política con 4672 acciones en una política con 1858 acciones
    :return: la matriz que convierte la política con 4672 acciones en una política con 1858 acciones
    """
    policy_matrix = torch.zeros((NUM_MOVES * ROWS * COLS, NUM_ACTIONS), dtype=torch.float32)

    for (move, row, col), action in TUPLE_TO_ACTION_MAP.items():
        policy_matrix[move * ROWS * COLS + row * COLS + col][action] = 1

    return policy_matrix


def add_promotion(uci_move: str, board: chess.Board) -> str:
    """
    Añade la promoción a un movimiento si es necesario
    :param uci_move: Movimiento en formato UCI
    :param board: Tablero de ajedrez
    :return: Movimiento en formato UCI corregido
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
