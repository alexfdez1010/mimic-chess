from typing import TextIO, List

from chess import pgn
import os

MINIMUM_NUMBER_OF_MOVES = 10  # Minimum number of moves to consider a game
NUM_GAMES_PER_FILE = 10000  # Maximum number of games per file


def game_is_valid_headers(headers: pgn.Headers, min_elo: int, max_elo: int) -> bool:
    """
    Checks whether a game is valid (there is time, a result and both players have theirs ELOs between the range)
    :param headers: game to check
    :param min_elo: minimum elo to filter the games
    :param max_elo: maximum elo to filter the games
    :return: true if the game is valid, false otherwise
    """

    if headers.get('TimeControl').find('+') == -1:
        return False

    if headers.get('Result') == '*':
        return False

    if headers.get('WhiteElo') == '?' or headers.get('BlackElo') == '?':
        return False

    white_elo = int(headers.get('WhiteElo'))
    black_elo = int(headers.get('BlackElo'))
    return min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo


def filter_by_minimum_number_of_moves(game: pgn.Game) -> bool:
    """
    Checks whether a game has at least MINIMUM_NUMBER_OF_MOVES moves
    :param game: game to check
    :return: true if the game has at least MINIMUM_NUMBER_OF_MOVES moves, false otherwise
    """
    temp_game = game

    for _ in range(MINIMUM_NUMBER_OF_MOVES):
        if (temp_game := temp_game.next()) is None:
            return False

    return True


def write_games(input_filename: str, f: TextIO, offsets: List[int], output_folder: str):
    """
    Writes the games in the output folder that have been selected in the previous step
    :param input_filename: name of the file to read
    :param f: file to read
    :param offsets: offsets of the games to write
    :param output_folder: folder to write the games
    """
    file_to_write = open(f"{output_folder}/{input_filename}_0.pgn", "w")

    for number_of_game, offset in enumerate(offsets, start=1):

        f.seek(offset)
        game = pgn.read_game(f)

        if not filter_by_minimum_number_of_moves(game):
            print(f"Game {number_of_game} has less than {MINIMUM_NUMBER_OF_MOVES} moves thus it is not valid")
            continue

        print(f"Game {number_of_game} is written: {game.headers.get('White')} {game.headers.get('Black')}")

        if number_of_game % NUM_GAMES_PER_FILE == 0:
            file_to_write.close()
            name_file = f"{output_folder}/{input_filename}_{number_of_game // NUM_GAMES_PER_FILE}.pgn"
            print(f"Changing to file {name_file} after writing {NUM_GAMES_PER_FILE} games")
            file_to_write = open(name_file, "w")

        file_to_write.write(f"{game}\n\n")

    if file_to_write is not None:
        file_to_write.close()


def get_offsets(file: TextIO, max_elo: int, min_elo: int):
    """
    Gets the offsets of the games that are valid
    :param file: file to read
    :param max_elo: maximum elo to filter the games
    :param min_elo: minimum elo to filter the games
    :return: offsets of the games that are valid
    """

    offsets = []
    number_of_games_read = 0

    while (offset := file.tell(), headers := pgn.read_headers(file))[1] is not None:

        number_of_games_read += 1

        if not game_is_valid_headers(headers, min_elo, max_elo):
            continue

        print(f"Game {number_of_games_read} is accepted: {headers.get('White')} {headers.get('Black')}")
        offsets.append(offset)

    return offsets


def filter_games(input_file: str, output_folder: str, min_elo: int, max_elo: int):
    """
    Filters the games in the input folder by the elo range and saves them in the output
    :param input_file: file that contains the games
    :param output_folder: folder that will contain the filtered games
    :param min_elo: minimum elo to filter the games
    :param max_elo: maximum elo to filter the games
    """

    if not input_file.endswith(".pgn"):
        print(f"Input file: {input_file} must be a pgn file")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, 'r') as f:

        offsets = get_offsets(f, max_elo, min_elo)
        filename = os.path.basename(input_file)
        filename = os.path.splitext(filename)[0]
        write_games(filename, f, offsets, output_folder)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='File that contains the games in pgn format')
    parser.add_argument('output_folder', type=str, help='Folder that will contain the filtered games')
    parser.add_argument('--min-elo', type=int, default=1000, help='Minimum elo to filter the games')
    parser.add_argument('--max-elo', type=int, default=3200, help='Maximum elo to filter the games')

    args = parser.parse_args()

    filter_games(args.input_file, args.output_folder, args.min_elo, args.max_elo)


if __name__ == '__main__':
    main()
