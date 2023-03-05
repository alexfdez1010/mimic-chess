import base64
from datetime import datetime

import chess
import streamlit as st
import chess.svg
import chess.pgn

from neural_network import NeuralNetworkProduction

PIECES = {
    "P": ("♙", 0, 8),
    "N": ("♘", 1, 2),
    "B": ("♗", 2, 2),
    "R": ("♖", 3, 2),
    "Q": ("♕", 4, 1),
    "K": ("♔", 5, 1),
    "p": ("♟", 6, 8),
    "n": ("♞", 7, 2),
    "b": ("♝", 8, 2),
    "r": ("♜", 9, 2),
    "q": ("♛", 10, 1),
    "k": ("♚", 11, 1),
}


@st.cache_resource
def get_app():
    return App()


class App:

    def __init__(self):
        self.board = chess.Board()
        self.initial_fen = chess.STARTING_FEN
        self.neural_network = NeuralNetworkProduction("test")

    def render(self):
        st.title("Mimic ♟️")
        st.write(
            "Mimic is the first model to take into account the time in modelling human decision-making in chess. "
            "The predictions will be influenced by the time left for both players and the increment for each move. "
            "Additionally, the model will output the expected time that the player will waste in the next move with "
            "the probabilities of playing each move and winning, losing or drawing the game. "
        )

        done = self.board.is_game_over(claim_draw=True)

        left_top, center_top, right_top = st.columns(3, gap="large")

        with right_top:

            uci_move = st.text_input(
                "Move in UCI format",
                "e2e4",
                help="Source square, destination square and promotion piece (if necessary). "
                     "Examples: c7c5, e7e8q (for the promotion specify the first letter in lowercase)."
            )

            if st.button("Move", help="Executes the move specify in the input box (e2e4)"):

                if done:
                    st.error("The game is over. You can't make any more moves.")
                else:
                    try:
                        self.board.push_uci(uci_move)
                    except ValueError:
                        st.error("Invalid move")

            if st.button("Undo", help="Undo the last move."):
                try:
                    self.board.pop()
                except IndexError:
                    st.error("There are no moves to undo.")

            if st.button("Restart", help="Restarts the game to the initial position."):
                self.board = chess.Board()
                self.initial_fen = chess.STARTING_FEN

            analyze = st.button("Analyze AI", help="Analyzes the position using the AI.")

        with left_top:
            time_self = st.text_input(
                "Time for the player to move",
                "00:01:00",
                placeholder="hh:mm:ss",
                help="Time left for the player to move (hh:mm:ss)",
                max_chars=8
            )

            time_opponent = st.text_input(
                "Time for the opponent to move",
                "00:01:00",
                placeholder="hh:mm:ss",
                help="Time left for the opponent to move (hh:mm:ss)",
                max_chars=8
            )

            increment = int(st.text_input(
                "Increment for both players",
                "1",
                placeholder="seconds",
                help="Time in seconds added to both players after each move",
                max_chars=2
            ))

        with center_top:
            color_white_squares, color_black_squares, color_margin, flip_board = self.render_sidebar()
            self.render_board(color_white_squares, color_black_squares, color_margin, flip_board)
            self.render_capture_pieces()

        if analyze:
            left_bottom, right_bottom = st.columns(2, gap="large")
            position_information = self.neural_network.predict(self.board.fen(), time_self, time_opponent, increment)

            with left_bottom:
                moves_string = "## Moves proposed in UCI format:\n\n"

                for move, probability in sorted(position_information.moves.items(), key=lambda x: x[1], reverse=True):
                    moves_string += f"- {move}: {100 * probability:.2f}%\n"

                st.write(moves_string)

            with right_bottom:
                st.write("## Other information of the position:\n\n")
                st.write(f"Probability of white winning: {100 * position_information.white_wining_probability:.2f}%")
                st.write(f"Probability of draw: {100 * position_information.draw_probability:.2f}%")
                st.write(f"Probability of black winning: {100 * position_information.black_wining_probability:.2f}%")
                st.write(f"Time expected to move: {position_information.time}")

        with st.expander("FEN"):
            st.write(
                "This section shows the current state of the board in FEN format."
                "The FEN format is a standard format to represent a chess position."
                "The box below shows the current FEN of the board. You can change it "
                "and the board will be updated after clicking the button below."
            )
            fen = st.text_input("FEN", self.board.fen())

            if st.button("Update FEN"):
                self.board.set_fen(fen)
                self.initial_fen = fen

        with st.expander("PGN"):

            st.write(
                "This section shows the game in PGN format."
                "The PGN format is a standard format to represent a complete chess game."
                "You can download the game in PGN format by clicking the button below."
            )
            pgn = self.create_pgn()

            st.write(pgn)
            st.download_button("Download PGN", pgn, "game.pgn", "text/plain")

    @staticmethod
    def render_svg(svg: str):
        """
        Render a SVG in Streamlit
        :param svg: SVG to render
        """
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

        css = '<p style="text-align:center; display: flex; justify-content: left; align-items: flex-start;">'
        html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'
        st.write(html, unsafe_allow_html=True)

    def render_board(self,
                     color_white_squares: str,
                     color_black_squares: str,
                     color_margin: str,
                     flip_board: bool = False):
        """
        Render a chess board in Streamlit
        :param color_white_squares: Color of the white squares
        :param color_black_squares: Color of the black squares
        :param color_margin: Color of the margin of the board
        :param flip_board: If the board should be flipped or not
        """
        self.render_svg(chess.svg.board(
            board=self.board,
            size=325,
            colors={
                "square light": color_white_squares,
                "square dark": color_black_squares,
                "margin": color_margin,
            },
            coordinates=True,
            flipped=flip_board,
            lastmove=None if len(self.board.move_stack) == 0 else self.board.peek()
        ))

    @staticmethod
    def render_sidebar():
        """Shows the sidebar in Streamlit"""
        with st.sidebar:
            st.image("resources/chessboard.png", use_column_width=True)
            st.title("Visual configuration of the board")

            columns = st.columns(2)

            with columns[0]:
                color_white_squares = st.color_picker("Light squares", "#FFFFFF", help="Color of the light squares")
                color_margin = st.color_picker("Margins", "#777777", help="Color of the margins of the board")

            with columns[1]:
                color_black_squares = st.color_picker("Dark squares", "#777777",
                                                      help="Color of the dark squares")
                flip_board = st.selectbox("Perspective", ["White", "Black"],
                                          help="If the board should be flipped or not")
                flip_board = flip_board == "Black"

        return color_white_squares, color_black_squares, color_margin, flip_board

    def create_pgn(self) -> str:
        """
        Creates a PGN string from the current game
        :return: PGN string
        """
        game = chess.pgn.Game()
        if self.initial_fen != chess.STARTING_FEN:
            game.setup(self.initial_fen)

        node = game
        for move in self.board.move_stack:
            node = node.add_variation(move)

        game.headers["Result"] = self.board.result()
        game.headers["White"] = "AI/Human"
        game.headers["Black"] = "AI/Human"
        game.headers["Date"] = datetime.now().strftime("%d.%m.%Y")
        game.headers["Event"] = "Mimic"
        game.headers["Site"] = "Mimic webapp"
        game.headers["Round"] = "1"

        return str(game)

    def render_capture_pieces(self):
        """Shows the captured pieces in Streamlit"""
        fen = self.board.fen()
        pieces_fen = fen.split(" ")[0]

        counter = [0 for _ in range(len(PIECES))]

        for piece in pieces_fen:
            if piece in PIECES:
                counter[PIECES[piece][1]] += 1

        white_capture_pieces = ""
        black_capture_pieces = ""

        for piece_emoji, index, piece_initial_count in PIECES.values():
            captured_pieces = piece_initial_count - counter[index]

            if captured_pieces > 0:
                if index < 6:
                    white_capture_pieces += f"{piece_emoji}{captured_pieces} "
                else:
                    black_capture_pieces += f"{piece_emoji}{captured_pieces} "

        st.markdown(f"### {white_capture_pieces}")
        st.markdown(f"### {black_capture_pieces}")
