def convert_ch(ch: chr) -> chr:
    """
    Si es una letra la convierte en minúscula/mayúscula si es mayúscula/minúscula en caso contrario no hace nada
    :param ch: Carácter a convertir
    :return: Carácter convertido
    """
    if ch.isalpha():
        return ch.lower() if ch.isupper() else ch.upper()
    return ch


def flip_fen(fen: str) -> str:
    """
    Devuelve la cadena FEN invertida
    :param fen: Cadena en formato FEN
    :return: Cadena en formato FEN invertida (se intercambian las piezas de color).
    """
    board, color, castles, en_passant, _, _ = fen.split(' ')
    new_board = ''

    for row in board[::-1].split('/'):
        row = ''.join(list(map(convert_ch, row)))
        new_board += row[::-1] + '/'

    new_board = new_board[:-1]

    if castles != '-':
        white_castles = castles.rstrip("kq")
        black_castles = castles.lstrip("KQ")
        new_castles = black_castles.upper() + white_castles.lower()
    else:
        new_castles = castles

    new_en_passant = f'{en_passant[0]}{str(9 - int(en_passant[1]))}' if en_passant != '-' else '-'

    return ' '.join([new_board, color, new_castles, new_en_passant, '0', '0'])


def flip_uci(uci: str) -> str:
    """
    Devuelve la cadena UCI invertida correspondiente al movimiento
    :param uci: Cadena en formato UCI
    :return: Cadena en formato UCI invertida (la misma jugada, pero las blancas son las negras y viceversa)
    """
    origin = uci[:2]
    dest = uci[2:4]

    origin = f'{origin[0]}{str(9 - int(origin[1]))}'
    dest = f'{dest[0]}{str(9 - int(dest[1]))}'

    promotion = uci[4:] if len(uci) == 5 else ''
    return origin + dest + promotion
