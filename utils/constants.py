"""Constantes comunes a los archivos de código"""

CHANNELS = 18  # Número de canales del tensor
ROWS, COLS = 8, 8  # Dimensiones del tablero de ajedrez
NUM_MOVES: int = 73  # Números de movimientos posibles desde una casilla concreta

NUM_ACTIONS: int = 1858  # Número de acciones totales

REWARD_ILLEGAL_MOVE: float = -1  # Recompensa por realizar una jugada ilegal
REWARD_WIN: float = 1  # Recompensa por ganar
REWARD_DRAW: float = 0  # Recompensa por empatar
REWARD_LOSS: float = -1  # Recompensa por perder

CHECKPOINT_PTH = 'checkpoint_supervised.pth'  # Lugar donde se almacena los últimos pesos de la red neuronal
CHECKPOINTS_DIRECTORY = 'checkpoints'  # Lugar donde se almacenan los pesos de las redes neuronales

ILLEGAL_MOVE_PENALTY = -1e10  # Penalización para las jugadas ilegales
