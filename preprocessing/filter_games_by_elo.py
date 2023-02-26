NUM_GAMES_PER_FILE = 20000


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help='Folder that contains the games in pgn format')
    parser.add_argument('output_folder', type=str, help='Folder that will contain the filtered games')
    parser.add_argument('--min-elo', type=int, default=0, help='Minimum elo to filter the games')
    parser.add_argument('--max-elo', type=int, default=3000, help='Maximum elo to filter the games')


if __name__ == '__main__':
    main()
