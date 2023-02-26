import torch
from torch import zeros, Tensor

from utils.constants import ROWS, COLS, CHANNELS
from utils.flip import flip_fen

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
