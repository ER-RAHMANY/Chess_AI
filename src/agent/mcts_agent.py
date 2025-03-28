# import random
# import math
# import chess

# # Node for the Monte Carlo Tree Search.


# class MCTSNode:
#     def __init__(self, state, parent=None, move=None):
#         """
#         state: A FEN string representing the board.
#         parent: The parent node.
#         move: The move (in UCI) that led from the parent to this node.
#         """
#         self.state = state
#         self.parent = parent
#         self.move = move  # The move that led to this node (None for root)
#         self.children = {}  # Dictionary: move (uci) -> MCTSNode
#         self.visits = 0
#         self.total_reward = 0.0
#         self.untried_moves = self.get_legal_moves()

#     def get_board(self):
#         return chess.Board(self.state)

#     def get_legal_moves(self):
#         board = self.get_board()
#         return [move.uci() for move in board.legal_moves]

#     def is_terminal(self):
#         board = self.get_board()
#         return board.is_game_over()

#     def best_child(self, c_param=math.sqrt(2)):
#         best_score = -float('inf')
#         best_child = None
#         for child in self.children.values():
#             if child.visits == 0:
#                 uct = float('inf')
#             else:
#                 uct = (child.total_reward / child.visits) + c_param * \
#                     math.sqrt(math.log(self.visits) / child.visits)
#             if uct > best_score:
#                 best_score = uct
#                 best_child = child
#         return best_child

#     def expand(self):
#         # Pick one move from the untried moves, remove it from the list, and add a new child.
#         move = self.untried_moves.pop()
#         board = self.get_board()
#         board.push(chess.Move.from_uci(move))
#         child_node = MCTSNode(state=board.fen(), parent=self, move=move)
#         self.children[move] = child_node
#         return child_node

#     def update(self, reward):
#         self.visits += 1
#         self.total_reward += reward


# def simulate_random_game(board):
#     """
#     Plays a random playout from the given board until game over.
#     Returns the total reward accumulated during the simulation.
#     Reward is based on capturing black pieces:
#         Pawn: +1, Knight: +3, Bishop: +3, Rook: +5, Queen: +9.
#     """
#     total_reward = 0
#     reward_mapping = {
#         chess.PAWN: 1,
#         chess.KNIGHT: 3,
#         chess.BISHOP: 3,
#         chess.ROOK: 5,
#         chess.QUEEN: 9,
#     }
#     board_copy = board.copy()
#     while not board_copy.is_game_over():
#         legal_moves = list(board_copy.legal_moves)
#         if not legal_moves:
#             break
#         move = random.choice(legal_moves)
#         reward = 0
#         if board_copy.is_capture(move):
#             if board_copy.is_en_passant(move):
#                 # Handle en passant capture.
#                 if board_copy.turn == chess.WHITE:
#                     captured_square = move.to_square - 8
#                 else:
#                     captured_square = move.to_square + 8
#             else:
#                 captured_square = move.to_square
#             captured_piece = board_copy.piece_at(captured_square)
#             if captured_piece and captured_piece.color == chess.BLACK:
#                 reward = reward_mapping.get(captured_piece.piece_type, 0)
#         board_copy.push(move)
#         total_reward += reward
#     return total_reward


# class MCTSAgent:
#     def __init__(self, iterations=1000):
#         """
#         iterations: The number of MCTS iterations to perform per move.
#         """
#         self.iterations = iterations

#     def select_move(self, state):
#         """
#         Given a state (FEN string), run MCTS for a fixed number of iterations and return the best move (in UCI).
#         """
#         root = MCTSNode(state)
#         for _ in range(self.iterations):
#             node = root
#             # SELECTION: Traverse the tree until reaching a node that is not fully expanded or terminal.
#             while node.untried_moves == [] and not node.is_terminal():
#                 node = node.best_child()
#             # EXPANSION: If node is non-terminal and has untried moves, expand one move.
#             if node.untried_moves and not node.is_terminal():
#                 node = node.expand()
#             # SIMULATION: Run a random playout from the node's state.
#             board = chess.Board(node.state)
#             reward = simulate_random_game(board)
#             # BACKPROPAGATION: Propagate the reward back through the tree.
#             while node is not None:
#                 node.update(reward)
#                 node = node.parent

#         # Return the move from the root with the highest visit count.
#         best_move = None
#         best_visits = -1
#         for move, child in root.children.items():
#             if child.visits > best_visits:
#                 best_visits = child.visits
#                 best_move = move
#         return best_move


# import random
# import math
# import chess


# class MCTSNode:e
#     def __init__(self, state, parent=None, move=None):
#         self.state = state
#         self.parent = parent
#         self.move = move
#         self.children = {}
#         self.visits = 0
#         self.total_reward = 0.0
#         self.untried_moves = self.get_legal_moves()

#     def get_board(self):
#         return chess.Board(self.state)

#     def get_legal_moves(self):
#         board = self.get_board()
#         return [move.uci() for move in board.legal_moves]

#     def is_terminal(self):
#         return self.get_board().is_game_over()

#     def best_child(self, c_param=math.sqrt(2)):
#         best_score = -float('inf')
#         best_child = None
#         for child in self.children.values():
#             if child.visits == 0:
#                 uct = float('inf')
#             else:
#                 exploitation = child.total_reward / child.visits
#                 exploration = c_param * \
#                     math.sqrt(math.log(self.visits) / child.visits)
#                 uct = exploitation + exploration
#             if uct > best_score:
#                 best_score = uct
#                 best_child = child
#         return best_child

#     def expand(self):
#         move = self.untried_moves.pop()
#         board = self.get_board()
#         board.push(chess.Move.from_uci(move))
#         child_node = MCTSNode(board.fen(), self, move)
#         self.children[move] = child_node
#         return child_node

#     def update(self, reward):
#         self.visits += 1
#         self.total_reward += reward


# def simulate_random_game(board):
#     total_reward = 0
#     reward_mapping = {chess.PAWN: 1, chess.KNIGHT: 3,
#                       chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
#     board_copy = board.copy()
#     # Joueur actuel au début de la simulation.
#     current_player = board_copy.turn

#     while not board_copy.is_game_over():
#         # Choisir un coup avec une heuristique.
#         move = choose_heuristic_move(board_copy)
#         reward = 0

#         if board_copy.is_capture(move):
#             if board_copy.is_en_passant(move):
#                 captured_square = move.to_square - \
#                     8 if board_copy.turn == chess.WHITE else move.to_square + 8
#             else:
#                 captured_square = move.to_square
#             captured_piece = board_copy.piece_at(captured_square)
#             if captured_piece and captured_piece.color != board_copy.turn:
#                 reward = reward_mapping.get(captured_piece.piece_type, 0)

#         board_copy.push(move)
#         total_reward += reward

#     # Récompense terminale.
#     if board_copy.is_checkmate():
#         total_reward += 100 if (board_copy.result() ==
#                                 "1-0" and current_player == chess.WHITE) else -100
#     elif board_copy.is_stalemate():
#         total_reward -= 10
#     return total_reward


# def choose_heuristic_move(board):
#     legal_moves = list(board.legal_moves)
#     # Heuristique : prioriser les captures et les échecs.
#     captures = [move for move in legal_moves if board.is_capture(move)]
#     checks = [move for move in legal_moves if board.gives_check(move)]
#     if checks:
#         return random.choice(checks)
#     if captures:
#         return random.choice(captures)
#     return random.choice(legal_moves)


# class MCTSAgent:
#     def __init__(self, iterations=1000):
#         self.iterations = iterations

#     def select_move(self, state):
#         root = MCTSNode(state)
#         for _ in range(self.iterations):
#             node = root
#             # Sélection
#             while node.untried_moves == [] and not node.is_terminal():
#                 node = node.best_child()
#             # Expansion
#             if node.untried_moves and not node.is_terminal():
#                 node = node.expand()
#             # Simulation
#             board = chess.Board(node.state)
#             current_player = board.turn
#             reward = simulate_random_game(board)
#             # Backpropagation avec ajustement pour le joueur.
#             while node is not None:
#                 node_player = chess.Board(node.state).turn
#                 adjusted_reward = reward if node_player == current_player else -reward
#                 node.update(adjusted_reward)
#                 node = node.parent
#         # Meilleur coup par visites.
#         return max(root.children.items(), key=lambda x: x[1].visits)[0]


# import torch
# import math
# import chess


# class MCTSNode:
#     def __init__(self, board, parent=None, prior=0.0):
#         self.board = board
#         self.parent = parent
#         self.children = {}
#         self.N = 0  # Visit count
#         self.W = 0  # Total value
#         self.Q = 0  # Mean value
#         self.P = prior  # Prior from NN

#     def is_leaf(self):
#         return len(self.children) == 0


# class MCTSAgent:
#     def __init__(self, model, simulations=100):
#         self.model = model
#         self.simulations = simulations

#     def select_move(self, env):
#         root_board = env.get_board().copy()
#         root = MCTSNode(root_board)

#         for _ in range(self.simulations):
#             node = root
#             path = [node]

#             # Selection
#             while not node.is_leaf() and not node.board.is_game_over():
#                 node = max(
#                     node.children.values(),
#                     key=lambda n: self.uct_score(path[-1].N, n)
#                 )
#                 path.append(node)

#             # Expansion and evaluation
#             if not node.board.is_game_over():
#                 self.expand_node(node)

#             # Backpropagation
#             value = node.Q
#             for node in reversed(path):
#                 node.N += 1
#                 node.W += value
#                 node.Q = node.W / node.N

#         # Select move with the most visits
#         move = max(root.children.items(), key=lambda item: item[1].N)[0]
#         return move.uci()

#     def expand_node(self, node):
#         board_tensor = self.board_to_tensor(node.board).unsqueeze(0)
#         with torch.no_grad():
#             policy_logits, value = self.model(board_tensor)
#             policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()
#             value = value.item()

#         for move in node.board.legal_moves:
#             move_idx = self.move_to_index(move)
#             new_board = node.board.copy()
#             new_board.push(move)
#             child = MCTSNode(new_board, parent=node, prior=policy[move_idx])
#             child.Q = value  # Initialize Q with network value
#             node.children[move] = child

#     def board_to_tensor(self, board):
#         tensor = torch.zeros(12, 8, 8)
#         piece_map = board.piece_map()
#         for square, piece in piece_map.items():
#             piece_type = piece.piece_type - 1  # pawn=0, ..., king=5
#             color_offset = 0 if piece.color == chess.WHITE else 6
#             row, col = divmod(square, 8)
#             tensor[color_offset + piece_type, 7 -
#                    row, col] = 1  # Flip row for white
#         return tensor

#     def move_to_index(self, move):
#         return move.from_square * 64 + move.to_square

#     def uct_score(self, parent_N, child, c_puct=1.0):
#         if child.N == 0:
#             return float('inf')
#         return child.Q + c_puct * child.P * math.sqrt(parent_N) / (1 + child.N)
import math
import chess
import torch

# --------------------------
# MCTS Node using FEN strings
# --------------------------


class MCTSNode:
    def __init__(self, state, parent=None, move=None, prior=0.0):
        """
        state: FEN string.
        parent: parent node.
        move: move (UCI string) that led to this state.
        prior: prior probability (from NN) assigned to this move.
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}  # dict mapping move (UCI string) to MCTSNode
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves = self.get_legal_moves()  # list of UCI moves
        self.prior = prior

    def get_board(self):
        return chess.Board(self.state)

    def get_legal_moves(self):
        board = self.get_board()
        return [move.uci() for move in board.legal_moves]

    def is_terminal(self):
        return self.get_board().is_game_over()

    def best_child(self, c_param):
        best_score = -float('inf')
        best_child = None
        # Use UCT with prior probabilities:
        for child in self.children.values():
            if child.visits == 0:
                uct = float('inf')
            else:
                exploitation = child.total_reward / child.visits
                exploration = c_param * child.prior * \
                    math.sqrt(self.visits) / (1 + child.visits)
                uct = exploitation + exploration
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward

# --------------------------
# MCTS Agent with NN integration
# --------------------------


class MCTSAgent:
    def __init__(self, model, iterations=1000, c_param=1.0):
        """
        model: a PyTorch neural network that takes a board tensor and outputs (policy_logits, value)
        iterations: number of MCTS iterations to run
        c_param: exploration constant for UCT
        """
        self.model = model
        self.iterations = iterations
        self.c_param = c_param

    def board_to_tensor(self, board):
        """
        Converts a chess.Board to a 12x8x8 tensor.
        Planes 0-5 for white pieces, 6-11 for black pieces.
        """
        tensor = torch.zeros(12, 8, 8)
        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            piece_type = piece.piece_type - 1  # pawn=0, knight=1, ..., king=5
            color_offset = 0 if piece.color == chess.WHITE else 6
            row, col = divmod(square, 8)
            # flip rows so rank8 is at top
            tensor[color_offset + piece_type, 7 - row, col] = 1
        return tensor

    def move_to_index(self, move):
        """
        Maps a move (UCI string) to a unique index (0-4095).
        This assumes a simple mapping: from_square * 64 + to_square.
        (Promotion moves and castling may require a more sophisticated mapping.)
        """
        move_obj = chess.Move.from_uci(move)
        return move_obj.from_square * 64 + move_obj.to_square

    def evaluate_state(self, state):
        """
        Uses the NN to evaluate a state.
        Returns:
            policy: a numpy array of shape (4096,) with move probabilities.
            value: a scalar value (typically between -1 and 1).
        """
        board = chess.Board(state)
        tensor = self.board_to_tensor(
            board).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            policy_logits, value = self.model(tensor)
            policy = torch.softmax(
                policy_logits, dim=1).squeeze().numpy()  # shape: (4096,)
            value = value.item()
        return policy, value

    def select_move(self, state):
        """
        Runs MCTS from the given state (FEN string) and returns the best move (UCI string).
        """
        root = MCTSNode(state)

        # If the root is non-terminal, expand it using the NN evaluation.
        if not root.is_terminal():
            policy, _ = self.evaluate_state(root.state)
            for move in root.untried_moves:
                board = root.get_board()
                board.push(chess.Move.from_uci(move))
                prior = policy[self.move_to_index(move)]
                child_node = MCTSNode(
                    board.fen(), parent=root, move=move, prior=prior)
                root.children[move] = child_node
            root.untried_moves = []  # fully expanded

        # Determine the current player at the root.
        current_player = chess.Board(root.state).turn

        for _ in range(self.iterations):
            node = root
            path = [node]

            # Selection: traverse until a node with untried moves or a terminal node.
            while node.untried_moves == [] and not node.is_terminal():
                node = node.best_child(self.c_param)
                path.append(node)

            # Expansion & Evaluation:
            if not node.is_terminal():
                # Expand one untried move if available.
                if node.untried_moves:
                    move = node.untried_moves.pop()
                    board = node.get_board()
                    board.push(chess.Move.from_uci(move))
                    new_state = board.fen()
                    policy, value = self.evaluate_state(new_state)
                    prior = policy[self.move_to_index(move)]
                    child_node = MCTSNode(
                        new_state, parent=node, move=move, prior=prior)
                    node.children[move] = child_node
                    node = child_node
                    path.append(node)
                    leaf_value = value
                else:
                    # Node is fully expanded but non-terminal; use evaluation.
                    _, leaf_value = self.evaluate_state(node.state)
            else:
                # Terminal node: assign terminal reward.
                board = node.get_board()
                if board.is_checkmate():
                    # If node.turn equals current_player then that player is about to move in a checkmated position.
                    leaf_value = -100 if board.turn == current_player else 100
                elif board.is_stalemate():
                    leaf_value = -10
                else:
                    leaf_value = 0

            # Backpropagation: update nodes along the path.
            for node in reversed(path):
                node_player = chess.Board(node.state).turn
                adjusted_value = leaf_value if node_player == current_player else -leaf_value
                node.update(adjusted_value)

        # After all iterations, choose the move from the root with the most visits.
        best_move, best_child = max(
            root.children.items(), key=lambda item: item[1].visits)
        return best_move
