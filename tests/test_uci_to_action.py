import chess

from utils.constants import ROWS, COLS
from utils.uci_to_action import valid_square, uci_to_tuple, create_maps, \
    uci_to_action, action_to_uci, add_promotion, create_policy_matrix


def test_valid_square():
    assert valid_square(0, 0)
    assert valid_square(7, 7)
    assert not valid_square(-1, 0)
    assert not valid_square(0, -1)
    assert not valid_square(8, 0)
    assert not valid_square(0, 8)


def test_uci_to_tuple():
    assert uci_to_tuple('e2e4') == (1, 1, 4)
    assert uci_to_tuple('e7e5') == (7 * 4 + 1, 6, 4)
    assert uci_to_tuple('e2e3') == (0, 1, 4)
    assert uci_to_tuple('e7e6') == (7 * 4 + 0, 6, 4)
    assert uci_to_tuple('g1f3') == (7 * 8 + 7, 0, 6)
    assert uci_to_tuple('b8c6') == (7 * 8 + 3, 7, 1)
    assert uci_to_tuple('e1g1') == (7 * 2 + 1, 0, 4)
    assert uci_to_tuple('e8c8') == (7 * 6 + 1, 7, 4)
    assert uci_to_tuple('d2d1n') == (64 + 1, 1, 3)
    assert uci_to_tuple('d2c1n') == (64, 1, 3)
    assert uci_to_tuple('d2e1n') == (64 + 2, 1, 3)
    assert uci_to_tuple('d2d1b') == (64 + 3 + 1, 1, 3)
    assert uci_to_tuple('d2c1b') == (64 + 3, 1, 3)
    assert uci_to_tuple('d2e1b') == (64 + 3 + 2, 1, 3)
    assert uci_to_tuple('d7c8r') == (64 + 3 * 2, 6, 3)
    assert uci_to_tuple('d7d8r') == (64 + 3 * 2 + 1, 6, 3)
    assert uci_to_tuple('d7e8r') == (64 + 3 * 2 + 2, 6, 3)
    assert uci_to_tuple('e4a4') == (7 * 6 + 3, 3, 4)
    assert uci_to_tuple('e4b4') == (7 * 6 + 2, 3, 4)
    assert uci_to_tuple('e4c4') == (7 * 6 + 1, 3, 4)
    assert uci_to_tuple('e4d4') == (7 * 6 + 0, 3, 4)
    assert uci_to_tuple('e4b1') == (7 * 5 + 2, 3, 4)


def test_tuple_to_action_and_action_to_tuple():
    tuple_to_action_map, action_to_tuple_map = create_maps()

    for key, value in tuple_to_action_map.items():
        assert action_to_tuple_map[value] == key

    for key, value in action_to_tuple_map.items():
        assert tuple_to_action_map[value] == key


def test_uci_to_action():
    assert uci_to_action('e2e4') == 68
    assert uci_to_action('e7e8n') == 1803
    assert uci_to_action('e7e8q') == 52
    assert uci_to_action('e7d8r') == 1839
    assert uci_to_action('e7f8b') == 1833
    assert uci_to_action('g1f3') == 1755
    assert uci_to_action('e7e8q') == 52
    assert uci_to_action('e7e8r') == 1847
    assert uci_to_action('e7e8b') == 1825
    assert uci_to_action('e7e8n') == 1803

    assert 'e2e3' == action_to_uci(uci_to_action('e2e3'))
    assert 'e7e8n' == action_to_uci(uci_to_action('e7e8n'))
    assert 'e7e8' == action_to_uci(uci_to_action('e7e8'))

    for i in range(1858):
        assert uci_to_action(action_to_uci(i)) == i


def test_add_promotion():
    board = chess.Board("rnb1k2r/pppp2Pp/8/8/7q/b7/PPP1PPPP/RNBQKBNR w KQkq - 1 5")
    assert add_promotion("g7g8", board) == 'g7g8q'
    assert add_promotion("g7g8n", board) == 'g7g8n'
    assert add_promotion("g7g8b", board) == 'g7g8b'
    assert add_promotion("g7g8r", board) == 'g7g8r'

    assert add_promotion("g7h8", board) == 'g7h8q'
    assert add_promotion("g7h8n", board) == 'g7h8n'
    assert add_promotion("g7h8b", board) == 'g7h8b'
    assert add_promotion("g7h8r", board) == 'g7h8r'


def test_create_policy_matrix():
    policy_matrix = create_policy_matrix()

    action_to_tuple_map = create_maps()[1]

    for action in range(1858):
        move, row, col = action_to_tuple_map[action]
        assert policy_matrix[move * ROWS * COLS + row * COLS + col, action] == 1
        assert policy_matrix[:, action].count_nonzero() == 1
