"""Common constants used in the project"""

CHANNELS = 18  # Number of channels in the input tensor
ROWS, COLS = 8, 8  # Number of rows and columns in the chess board
NUM_MOVES: int = 73  # Number of possible moves from a square

NUM_ACTIONS: int = 1858  # Number of possible actions

CHECKPOINTS_DIRECTORY = 'checkpoints'  # Directory where the checkpoints are saved

ILLEGAL_MOVE_PENALTY = -1e10  # Penalty for illegal moves
