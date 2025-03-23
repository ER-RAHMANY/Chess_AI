import chess


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        """
        Resets the board to the initial position.
        Returns:
            state (str): The board state in FEN notation.
        """
        self.board.reset()
        return self.get_state()

    def get_state(self):
        """
        Returns the current state of the board.
        Here we use the FEN string for simplicity.
        Returns:
            state (str): The board state in FEN notation.
        """
        return self.board.fen()

    def get_board(self):

        return self.board

    def step(self, move):
        """
        Applies a move to the board and computes a reward based on capturing black pieces.
        Args:
            move (str): The move in UCI format (e.g., "e2e4").
        Returns:
            next_state (str): The updated board state in FEN.
            reward (float): Reward based on the capture (if any).
            done (bool): True if the game is over.
            info (dict): Additional info (empty for now).
        """
        try:
            chess_move = chess.Move.from_uci(move)
        except Exception as e:
            raise ValueError(f"Invalid move format '{move}'. Error: {e}")

        if chess_move not in self.board.legal_moves:
            raise ValueError(f"Illegal move attempted: {move}")

        # Initialize reward.
        reward = 0

        # Check if the move is a capture before applying it.
        if self.board.is_capture(chess_move):
            # Determine the square of the captured piece.
            if self.board.is_en_passant(chess_move):
                # For en passant, the captured pawn is not on the destination square.
                if self.board.turn == chess.WHITE:
                    captured_square = chess_move.to_square - 8
                else:
                    captured_square = chess_move.to_square + 8
            else:
                captured_square = chess_move.to_square

            captured_piece = self.board.piece_at(captured_square)
            # Only assign a reward if a black piece is captured.
            if captured_piece is not None and captured_piece.color == chess.WHITE:
                reward_mapping = {
                    chess.PAWN: 1,
                    chess.KNIGHT: 3,
                    chess.BISHOP: 3,
                    chess.ROOK: 5,
                    chess.QUEEN: 9,
                }
                reward = reward_mapping.get(captured_piece.piece_type, 0)

        # Apply the move.
        self.board.push(chess_move)
        done = self.board.is_game_over()

        return self.get_state(), reward, done, {}

    # def step(self, move):
    #     """
    #     Applies a move to the board.
    #     Args:
    #         move (str): The move in UCI format (e.g., "e2e4").
    #     Returns:
    #         next_state (str): The updated board state in FEN.
    #         reward (float): Reward signal (only given at game end).
    #         done (bool): True if the game is over.
    #         info (dict): Additional info (empty for now).
    #     """
    #     try:
    #         chess_move = chess.Move.from_uci(move)
    #     except Exception as e:
    #         raise ValueError(f"Invalid move format '{move}'. Error: {e}")

    #     if chess_move not in self.board.legal_moves:
    #         raise ValueError(f"Illegal move attempted: {move}")

    #     self.board.push(chess_move)
    #     done = self.board.is_game_over()
    #     reward = 0
    #     if done:
    #         result = self.board.result()  # "1-0", "0-1", or "1/2-1/2"
    #         if result == "1-0":
    #             reward = 1    # White wins
    #         elif result == "0-1":
    #             reward = -1   # Black wins
    #         else:
    #             reward = 0.5  # Draw reward
    #     return self.get_state(), reward, done, {}

    # def render(self):
    #     """
    #     Renders the current board to the console.
    #     """
    #     print(self.board)
