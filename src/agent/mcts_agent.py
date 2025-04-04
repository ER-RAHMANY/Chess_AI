
import math
import chess
import torch

class MCTSNode:
    def __init__(self, state, parent=None, move=None, prior=0.0, board=None):
        """
        state: FEN string.
        parent: parent node.
        move: move (UCI string) that led to this state.
        prior: prior probability (from NN) assigned to this move.
        """
        self.board = board if board else chess.Board(state)
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}  # dict mapping move (UCI string) to MCTSNode
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves = self.get_legal_moves()  # list of UCI moves
        self.prior = prior
        
    def get_board(self):
        return self.board

    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]

    def is_terminal(self):
        return self.board.is_game_over()

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
        root = MCTSNode(state, board=chess.Board(state))  # cache board

        # If the root is non-terminal, expand it using the NN evaluation.
        if not root.is_terminal():
            policy, _ = self.evaluate_state(root.state)
            for move in root.untried_moves:
                board = root.get_board().copy()
                board.push(chess.Move.from_uci(move))
                prior = policy[self.move_to_index(move)]
                child_node = MCTSNode(
                    board.fen(), parent=root, move=move, prior=prior, board=board
                )
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
                    board = node.get_board().copy()
                    board.push(chess.Move.from_uci(move))
                    new_state = board.fen()
                    policy, value = self.evaluate_state(new_state)
                    prior = policy[self.move_to_index(move)]
                    child_node = MCTSNode(
                        new_state, parent=node, move=move, prior=prior, board=board)
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
                    leaf_value = -1 if board.turn == current_player else 1
                elif board.is_stalemate():
                    leaf_value = -0.5
                else:
                    leaf_value = 0

            # Backpropagation: update nodes along the path.
            for node in reversed(path):
                node_player = chess.Board(node.state).turn
                adjusted_value = leaf_value if node_player == current_player else -leaf_value
                print(adjusted_value)
                node.update(adjusted_value)

        # After all iterations, choose the move from the root with the most visits.
        best_move, best_child = max(
            root.children.items(), key=lambda item: item[1].visits)
        return best_move