from utils.constants import CHANNELS, ROWS, COLS
from utils.fen_to_tensor import fen_to_tensor, PIECE_TO_CHANNEL, CASTLE_TO_CHANNEL, COLOUR, PADDED_CONVOLUTION


def test_pieces():
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    tensor = fen_to_tensor(fen)
    assert tensor.shape == (CHANNELS, ROWS, COLS)
    assert all(tensor[PIECE_TO_CHANNEL['P'], 1] == 1)
    assert tensor[PIECE_TO_CHANNEL['N'], 0, 1] == 1
    assert tensor[PIECE_TO_CHANNEL['B'], 0, 2] == 1
    assert tensor[PIECE_TO_CHANNEL['R'], 0, 0] == 1
    assert tensor[PIECE_TO_CHANNEL['Q'], 0, 3] == 1
    assert tensor[PIECE_TO_CHANNEL['K'], 0, 4] == 1
    assert all(tensor[PIECE_TO_CHANNEL['p'], 6] == 1)
    assert tensor[PIECE_TO_CHANNEL['n'], 7, 1] == 1
    assert tensor[PIECE_TO_CHANNEL['b'], 7, 2] == 1
    assert tensor[PIECE_TO_CHANNEL['r'], 7, 0] == 1
    assert tensor[PIECE_TO_CHANNEL['q'], 7, 3] == 1
    assert tensor[PIECE_TO_CHANNEL['k'], 7, 4] == 1

    fen = 'rnbqkbnr/p1pppppp/8/1pP5/8/8/PP1PPPPP/RNBQKBNR w KQkq b6 0 1'
    tensor = fen_to_tensor(fen)
    assert tensor[PIECE_TO_CHANNEL['p'], 7, 1] == 1
    assert tensor[PIECE_TO_CHANNEL['p'], 4, 1] == 0

    fen = 'rnbqkbnr/p1pppppp/8/8/1pP5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1'
    tensor = fen_to_tensor(fen)
    assert tensor[PIECE_TO_CHANNEL['P'], 0, 2] == 1
    assert tensor[PIECE_TO_CHANNEL['P'], 3, 2] == 0


def test_castling_rights():
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    tensor = fen_to_tensor(fen)

    assert (tensor[CASTLE_TO_CHANNEL['K']] == 1).all()
    assert (tensor[CASTLE_TO_CHANNEL['Q']] == 1).all()
    assert (tensor[CASTLE_TO_CHANNEL['k']] == 1).all()
    assert (tensor[CASTLE_TO_CHANNEL['q']] == 1).all()

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1'
    tensor = fen_to_tensor(fen)

    assert (tensor[CASTLE_TO_CHANNEL['K']] == 1).all()
    assert (tensor[CASTLE_TO_CHANNEL['Q']] == 1).all()
    assert (tensor[CASTLE_TO_CHANNEL['k']] == 1).all()
    assert (tensor[CASTLE_TO_CHANNEL['q']] == 0).all()

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1'
    tensor = fen_to_tensor(fen)

    assert (tensor[CASTLE_TO_CHANNEL['K']] == 0).all()
    assert (tensor[CASTLE_TO_CHANNEL['Q']] == 0).all()
    assert (tensor[CASTLE_TO_CHANNEL['k']] == 0).all()
    assert (tensor[CASTLE_TO_CHANNEL['q']] == 0).all()


def test_colour_and_convolution():
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    tensor = fen_to_tensor(fen)
    assert (tensor[COLOUR] == 0).all()
    assert (tensor[PADDED_CONVOLUTION] == 1).all()

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1'
    tensor = fen_to_tensor(fen)
    assert (tensor[COLOUR] == 1).all()
    assert (tensor[PADDED_CONVOLUTION] == 1).all()
