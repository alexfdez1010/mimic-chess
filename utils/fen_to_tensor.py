from torch import zeros, Tensor

from utils.constants import ROWS, COLS, CHANNELS

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
}  # correspondencias entre las piezas y los canales

CASTLE_TO_CHANNEL = {
    'K': 12,
    'Q': 13,
    'k': 14,
    'q': 15
}  # correspondencias entre los enroques disponibles y los canales

COLOUR = 16

PADDED_CONVOLUTION = 17  # Posición del canal de ayuda para facilitar la convolución


def fen_to_tensor(fen: str) -> Tensor:
    """
    Convierte una cadena en formato FEN en el tensor correspondiente para la red neuronal
    :param fen: una cadena en formato FEN
    :return: tensor para la red neuronal
    """
    tensor = zeros(CHANNELS, ROWS, COLS)
    tensor[PADDED_CONVOLUTION] = 1

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
