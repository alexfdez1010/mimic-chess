import os
from random import random

from chess import pgn

from utils.constants import VICTORY, DRAW, LOSS
from utils.time_utils import time_string_to_seconds

POSITIONS_PER_FILE = 500000  # Maximum number of positions per file

RESULT_TO_INT = {
    "1-0": VICTORY,
    "1/2-1/2": DRAW,
    "0-1": LOSS
}


def get_time_string(comment: str) -> str:
    """
    Gets the time string from a comment of a move
    :param comment: comment to parse
    :return: time string
    """
    clk_index = comment.find("%clk")
    return comment[clk_index + 5:clk_index + 12]


def create_csv_entry(game: pgn.Game, time_self: int, time_rival: int, increment: int, result: int,
                     time_used: int) -> str:
    """
    Creates a csv entry from a position
    :param game: game to convert
    :param time_self: time of the player
    :param time_rival: time of the opponent
    :param increment: increment of time for both players
    :param result: result of the game from the point of view of the player
    :param time_used: time used by the player in the last move
    :return: csv entry
    """

    move = game.next().move.uci()
    fen = game.board().fen()
    return f"{fen},{time_self},{time_rival},{increment},{move},{result},{time_used}\n"


def get_times(game: pgn.Game, increment: int) -> (int, int, int):
    """
    Gets the times of the players
    :param game: game to parse
    :param increment: increment of time for both players
    :return: a tuple with the time of the player, the time of the opponent and the time used by the player in the move
    """
    time_self = time_string_to_seconds(get_time_string(game.next().comment))

    if game.next().next() is None:
        time_rival = time_string_to_seconds(get_time_string(game.comment))
        return time_self, time_rival, 0

    time_rival = time_string_to_seconds(get_time_string(game.next().next().comment))

    if game.next().next().next() is None:
        return time_self, time_rival, 0

    time_after_move = time_string_to_seconds(get_time_string(game.next().next().next().comment))
    time_used = time_self + increment - time_after_move

    return time_self, time_rival, time_used


def games_to_positions(input_file: str,
                       output_folder_training: str,
                       output_folder_validation: str,
                       split: float = 0.95):
    """
    Converts a folder of games in pgn format to a folder of positions in pickle format
    :param input_file: file that contains the games in pgn format
    :param output_folder_training: folder that will contain the positions for training (in csv format)
    :param output_folder_validation: folder that will contain the positions for validation (in csv format)
    :param split: percentage of games to use for training
    """
    written_to_training = 0
    written_to_validation = 0
    num_games = 0

    base_filename = os.path.basename(input_file).split(".")[0]

    file_to_write_training = open(f"{output_folder_training}/{base_filename}_0.csv", "w")
    file_to_write_validation = open(f"{output_folder_validation}/{base_filename}_0.csv", "w")

    with open(input_file, 'r') as file:

        while game := pgn.read_game(file):

            num_games += 1

            to_training = random() < split
            file_to_write = file_to_write_training if to_training else file_to_write_validation

            print(f"{base_filename} - Processing game {num_games}: "
                  f"{game.headers['White']} vs {game.headers['Black']} "
                  f"to {'training' if to_training else 'validation'}")

            increment = int(game.headers["TimeControl"].split("+")[-1])
            result = RESULT_TO_INT[game.headers["Result"]]

            while game.next():
                written_to_training += 1 if to_training else 0
                written_to_validation += 1 if not to_training else 0

                time_self, time_rival, time_used = get_times(game, increment)
                file_to_write.write(create_csv_entry(game, time_self, time_rival, increment, result, time_used))

                game = game.next()
                result = 2 - result

                if to_training and written_to_training % POSITIONS_PER_FILE == 0:
                    file_to_write_training.close()
                    fn = f"{output_folder_training}/{base_filename}_{written_to_training // POSITIONS_PER_FILE}.csv"
                    file_to_write_training = open(fn, "w")
                    file_to_write = file_to_write_training

                if not to_training and written_to_validation % POSITIONS_PER_FILE == 0:
                    file_to_write_validation.close()
                    fn = f"{output_folder_validation}/{base_filename}_{written_to_validation // POSITIONS_PER_FILE}.csv"
                    file_to_write_validation = open(fn, "w")
                    file_to_write = file_to_write_validation


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Converts a folder of games in pgn format to a folder of positions in csv format'
    )
    parser.add_argument('input_file', type=str, help='File that contains the games in pgn format')
    parser.add_argument('output_folder', type=str, help='Folder that will contain the positions')
    parser.add_argument('--split', type=float, default=0.95, help='Percentage of games to use for training')

    args = parser.parse_args()

    training_folder = os.path.join(args.output_folder, "training")
    validation_folder = os.path.join(args.output_folder, "validation")

    games_to_positions(args.input_file, training_folder, validation_folder, args.split)


if __name__ == '__main__':
    main()
