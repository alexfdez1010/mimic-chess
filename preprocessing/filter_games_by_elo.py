from chess import pgn
import os

MINIMUM_NUMBER_OF_MOVES = 10  # Minimum number of moves to consider a game
NUM_GAMES_PER_FILE = 10000  # Maximum number of games per file


def game_is_valid(game: pgn.Game, min_elo: int, max_elo: int) -> bool:
    """
    Checks whether a game is valid (has at least 10 moves, there is time and both players have an elo)
    :param game: game to check
    :param min_elo: minimum elo to filter the games
    :param max_elo: maximum elo to filter the games
    :return: true if the game is valid, false otherwise
    """
    temp_game = game

    if game.headers['TimeControl'].find('+') == -1:
        return False

    for _ in range(MINIMUM_NUMBER_OF_MOVES):
        if temp_game.next() is None:
            return False
        temp_game = temp_game.next()

    if game.headers['WhiteElo'] == '?' or game.headers['BlackElo'] == '?':
        return False

    white_elo = int(game.headers['WhiteElo'])
    black_elo = int(game.headers['BlackElo'])
    return min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo


def filter_games(input_folder: str, output_folder: str, min_elo: int, max_elo: int):
    """
    Filters the games in the input folder by the elo range and saves them in the output
    :param input_folder: folder that contains the games in pgn files
    :param output_folder: folder that will contain the filtered games
    :param min_elo: minimum elo to filter the games
    :param max_elo: maximum elo to filter the games
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    number_of_games_written = 0
    number_of_games_read = 0
    file_to_write = open(f"{output_folder}/games_0.pgn", "w")

    for file in os.listdir(input_folder):

        if not file.endswith('.pgn'):
            continue

        with open(os.path.join(input_folder, file), 'r') as f:
            game = pgn.read_game(f)

            while game is not None:

                number_of_games_read += 1

                if not game_is_valid(game, min_elo, max_elo):
                    print(f"Game {number_of_games_read} is discarded: {game.headers['White']} {game.headers['Black']}")
                    game = pgn.read_game(f)
                    continue

                print(f"Game {number_of_games_read} is used: {game.headers['White']} {game.headers['Black']}")
                number_of_games_written += 1
                file_to_write.write(f"{str(game)}\n\n")
                game = pgn.read_game(f)

                if number_of_games_written % NUM_GAMES_PER_FILE == 0:
                    file_to_write.close()
                    name_file = f"{output_folder}/games_{number_of_games_written // NUM_GAMES_PER_FILE}.pgn"
                    print(f"Changing to file {name_file} after writing {number_of_games_written} games")
                    file_to_write = open(name_file, "w")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help='Folder that contains the games in pgn format')
    parser.add_argument('output_folder', type=str, help='Folder that will contain the filtered games')
    parser.add_argument('--min-elo', type=int, default=0, help='Minimum elo to filter the games')
    parser.add_argument('--max-elo', type=int, default=3000, help='Maximum elo to filter the games')

    args = parser.parse_args()

    filter_games(args.input_folder, args.output_folder, args.min_elo, args.max_elo)


if __name__ == '__main__':
    main()
