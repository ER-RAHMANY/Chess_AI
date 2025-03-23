import random
import math
import chess

# Node for the Monte Carlo Tree Search.


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        """
        state: A FEN string representing the board.
        parent: The parent node.
        move: The move (in UCI) that led from the parent to this node.
        """
        self.state = state
        self.parent = parent
        self.move = move  # The move that led to this node (None for root)
        self.children = {}  # Dictionary: move (uci) -> MCTSNode
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves = self.get_legal_moves()

    def get_board(self):
        return chess.Board(self.state)

    def get_legal_moves(self):
        board = self.get_board()
        return [move.uci() for move in board.legal_moves]

    def is_terminal(self):
        board = self.get_board()
        return board.is_game_over()

    def best_child(self, c_param=math.sqrt(2)):
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            if child.visits == 0:
                uct = float('inf')
            else:
                uct = (child.total_reward / child.visits) + c_param * \
                    math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child

    def expand(self):
        # Pick one move from the untried moves, remove it from the list, and add a new child.
        move = self.untried_moves.pop()
        board = self.get_board()
        board.push(chess.Move.from_uci(move))
        child_node = MCTSNode(state=board.fen(), parent=self, move=move)
        self.children[move] = child_node
        return child_node

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward


def simulate_random_game(board):
    """
    Plays a random playout from the given board until game over.
    Returns the total reward accumulated during the simulation.
    Reward is based on capturing black pieces:
        Pawn: +1, Knight: +3, Bishop: +3, Rook: +5, Queen: +9.
    """
    total_reward = 0
    reward_mapping = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    board_copy = board.copy()
    while not board_copy.is_game_over():
        legal_moves = list(board_copy.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        reward = 0
        if board_copy.is_capture(move):
            if board_copy.is_en_passant(move):
                # Handle en passant capture.
                if board_copy.turn == chess.WHITE:
                    captured_square = move.to_square - 8
                else:
                    captured_square = move.to_square + 8
            else:
                captured_square = move.to_square
            captured_piece = board_copy.piece_at(captured_square)
            if captured_piece and captured_piece.color == chess.BLACK:
                reward = reward_mapping.get(captured_piece.piece_type, 0)
        board_copy.push(move)
        total_reward += reward
    return total_reward


class MCTSAgent:
    def __init__(self, iterations=1000):
        """
        iterations: The number of MCTS iterations to perform per move.
        """
        self.iterations = iterations

    def select_move(self, state):
        """
        Given a state (FEN string), run MCTS for a fixed number of iterations and return the best move (in UCI).
        """
        root = MCTSNode(state)
        for _ in range(self.iterations):
            node = root
            # SELECTION: Traverse the tree until reaching a node that is not fully expanded or terminal.
            while node.untried_moves == [] and not node.is_terminal():
                node = node.best_child()
            # EXPANSION: If node is non-terminal and has untried moves, expand one move.
            if node.untried_moves and not node.is_terminal():
                node = node.expand()
            # SIMULATION: Run a random playout from the node's state.
            board = chess.Board(node.state)
            reward = simulate_random_game(board)
            # BACKPROPAGATION: Propagate the reward back through the tree.
            while node is not None:
                node.update(reward)
                node = node.parent

        # Return the move from the root with the highest visit count.
        best_move = None
        best_visits = -1
        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
        return best_move
