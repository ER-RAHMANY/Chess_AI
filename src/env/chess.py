# import chess


# class ChessEnv:
#     def __init__(self):
#         self.board = chess.Board()

#     def reset(self):
#         """
#         Resets the board to the initial position.
#         Returns:
#             state (str): The board state in FEN notation.
#         """
#         self.board.reset()
#         return self.get_state()

#     def get_state(self):
#         """
#         Returns the current state of the board.
#         Here we use the FEN string for simplicity.
#         Returns:
#             state (str): The board state in FEN notation.
#         """
#         return self.board.fen()

#     def get_board(self):

#         return self.board

#     def step(self, move):
#         """
#         Applies a move to the board and computes a reward based on capturing black pieces.
#         Args:
#             move (str): The move in UCI format (e.g., "e2e4").
#         Returns:
#             next_state (str): The updated board state in FEN.
#             reward (float): Reward based on the capture (if any).
#             done (bool): True if the game is over.
#             info (dict): Additional info (empty for now).
#         """
#         try:
#             chess_move = chess.Move.from_uci(move)
#         except Exception as e:
#             raise ValueError(f"Invalid move format '{move}'. Error: {e}")

#         if chess_move not in self.board.legal_moves:
#             raise ValueError(f"Illegal move attempted: {move}")

#         # Initialize reward.
#         reward = 0

#         # Check if the move is a capture before applying it.
#         if self.board.is_capture(chess_move):
#             # Determine the square of the captured piece.
#             if self.board.is_en_passant(chess_move):
#                 # For en passant, the captured pawn is not on the destination square.
#                 if self.board.turn == chess.BLACK:
#                     captured_square = chess_move.to_square - 8
#                 else:
#                     captured_square = chess_move.to_square + 8
#             else:
#                 captured_square = chess_move.to_square

#             captured_piece = self.board.piece_at(captured_square)
#             # Only assign a reward if a black piece is captured.
#             if captured_piece is not None and captured_piece.color == chess.BLACK:
#                 reward_mapping = {
#                     chess.PAWN: 1,
#                     chess.KNIGHT: 3,
#                     chess.BISHOP: 3,
#                     chess.ROOK: 5,
#                     chess.QUEEN: 9,
#                 }
#                 reward = reward_mapping.get(captured_piece.piece_type, 0)

#         # Apply the move.
#         self.board.push(chess_move)
#         done = self.board.is_game_over()

#         return self.get_state(), reward, done, {}

#     # def step(self, move):
#     #     """
#     #     Applies a move to the board.
#     #     Args:
#     #         move (str): The move in UCI format (e.g., "e2e4").
#     #     Returns:
#     #         next_state (str): The updated board state in FEN.
#     #         reward (float): Reward signal (only given at game end).
#     #         done (bool): True if the game is over.
#     #         info (dict): Additional info (empty for now).
#     #     """
#     #     try:
#     #         chess_move = chess.Move.from_uci(move)
#     #     except Exception as e:
#     #         raise ValueError(f"Invalid move format '{move}'. Error: {e}")

#     #     if chess_move not in self.board.legal_moves:
#     #         raise ValueError(f"Illegal move attempted: {move}")

#     #     self.board.push(chess_move)
#     #     done = self.board.is_game_over()
#     #     reward = 0
#     #     if done:
#     #         result = self.board.result()  # "1-0", "0-1", or "1/2-1/2"
#     #         if result == "1-0":
#     #             reward = 1    # White wins
#     #         elif result == "0-1":
#     #             reward = -1   # Black wins
#     #         else:
#     #             reward = 0.5  # Draw reward
#     #     return self.get_state(), reward, done, {}

#     # def render(self):
#     #     """
#     #     Renders the current board to the console.
#     #     """
#     #     print(self.board)


import chess


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        # Reward configuration for flexible adjustments.
        self.reward_config = {
            "legal_move": 0.1,  # Small reward for making a legal move.
            "capture_rewards": {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
            },
            # Large reward for checkmating the opponent.
            "checkmate_reward": 100,
            "checkmate_penalty": -100,  # Large penalty for being checkmated.
            "stalemate_penalty": -10,  # Penalty for stalemates.
            # Small reward for putting the opponent in check.
            "check_reward": 0.5,
            "promotion_reward": 5,  # Reward for promoting a pawn.
            # Reward for controlling central squares.
            "center_control_reward": 0.2,
        }
        # Central squares.
        self.central_squares = [chess.E4, chess.D4, chess.E5, chess.D5]

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
        """
        Returns the current board object.
        Returns:
            board (chess.Board): The current chess board.
        """
        return self.board

    def step(self, move):
        """
        Applies a move to the board and computes a reward based on the configured strategy.
        Args:
            move (str): The move in UCI format (e.g., "e2e4").
        Returns:
            next_state (str): The updated board state in FEN.
            reward (float): Reward based on the move.
            done (bool): True if the game is over.
            info (dict): Additional info.
        """
        try:
            chess_move = chess.Move.from_uci(move)
        except Exception as e:
            raise ValueError(f"Invalid move format '{move}'. Error: {e}")

        if chess_move not in self.board.legal_moves:
            raise ValueError(f"Illegal move attempted: {move}")

        # Initialize reward.
        reward = self.reward_config["legal_move"]

        # Check for captures.
        if self.board.is_capture(chess_move):
            captured_square = self._get_captured_square(chess_move)
            captured_piece = self.board.piece_at(captured_square)
            if captured_piece is not None and captured_piece.color == chess.WHITE:
                reward += self.reward_config["capture_rewards"].get(
                    captured_piece.piece_type, 0)

        # Check for promotions.
        if chess_move.promotion is not None:
            reward += self.reward_config["promotion_reward"]

        # Check for central control.
        if chess_move.to_square in self.central_squares:
            reward += self.reward_config["center_control_reward"]

        # Check for checks.
        if self.board.gives_check(chess_move):
            reward += self.reward_config["check_reward"]

        # Apply the move.
        self.board.push(chess_move)
        done = self.board.is_game_over()

        # Add rewards/penalties for terminal states.
        if done:
            if self.board.is_checkmate():
                reward += self.reward_config["checkmate_reward"] if self.board.turn == chess.BLACK else self.reward_config["checkmate_penalty"]
            elif self.board.is_stalemate():
                reward += self.reward_config["stalemate_penalty"]

        return self.get_state(), reward, done, {}

    def _get_captured_square(self, move):
        """
        Helper function to determine the square of the captured piece.
        Args:
            move (chess.Move): The move being applied.
        Returns:
            square (int): The square of the captured piece.
        """
        if self.board.is_en_passant(move):
            return move.to_square - 8 if self.board.turn == chess.BLACK else move.to_square + 8
        return move.to_square

    def render(self):
        """
        Renders the current board to the console.
        """
        print(self.board)
