import random
import chess

class BaseAgent:
    def __init__(self):
        pass

    def select_move(self, state):
        """
        Given a state (in FEN format), select the next move.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("select_move must be implemented by the agent.")

class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def select_move(self, state):
        """
        Converts the FEN state into a board, selects a random legal move,
        and returns the move in UCI format.
        Args:
            state (str): Board state in FEN notation.
        Returns:
            move (str): A legal move in UCI format.
        """
        board = chess.Board(state)
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves).uci()
        else:
            return None  # No legal moves available (game over)
