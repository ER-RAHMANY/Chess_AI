import os
import torch
import chess.pgn
import chess.engine
from torch.utils.data import Dataset
from tqdm import tqdm

STOCKFISH_PATH = "src/fairy-stockfish_x86-64.exe"  # or Fairy-Stockfish
MAX_CP = 1000  # Max centipawn value to clip for scaling

def scale_cp(cp_score):
    """Scales centipawn score to [-1, 1]."""
    cp = cp_score / 100
    cp = 1 / (1 + 10 ** (-cp / 4))
    return 2 * cp - 1  # scale to [-1, 1]

def board_to_tensor(board):
    """Converts a python-chess board to a 8x8x12 tensor."""
    tensor = torch.zeros(12, 8, 8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1  # 0–5
        color_offset = 0 if piece.color == chess.WHITE else 6
        row, col = divmod(square, 8)
        tensor[color_offset + piece_type, 7 - row, col] = 1
    return tensor

def board_to_state(board):
    """Converts a python-chess board to a tensor of shape (19, 8, 8)."""
    tensor = torch.zeros(19, 8, 8, dtype=torch.float32)

    # Plane 0: Turn
    tensor[0, :, :] = board.turn

    # Planes 1-4: Castling rights
    tensor[1, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[2, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[3, :, :] = board.has_queenside_castling_rights(chess.BLACK)
    tensor[4, :, :] = board.has_kingside_castling_rights(chess.BLACK)

    # Plane 5: Fifty-move repetition
    tensor[5, :, :] = board.can_claim_fifty_moves()

    # Planes 6-17: Piece positions
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1  # 0–5
        color_offset = 6 if piece.color == chess.WHITE else 12
        row, col = divmod(square, 8)
        tensor[color_offset + piece_type, 7 - row, col] = 1

    # Plane 18: En passant square
    if board.ep_square is not None:
        ep_row, ep_col = divmod(board.ep_square, 8)
        tensor[18, 7 - ep_row, ep_col] = 1

    return tensor

def move_to_index(move):
    """Maps a move to an index (simplified version)."""
    return move.from_square * 64 + move.to_square  # 0–4095

def move_to_index(move):
    """Maps a move (including promotions explicitly) to a unique index."""
    from_sq, to_sq = move.from_square, move.to_square

    # Check for promotion explicitly
    if move.promotion:
        promo_piece_to_idx = {chess.QUEEN:0, chess.ROOK:1, chess.BISHOP:2, chess.KNIGHT:3}
        promo_offset = promo_piece_to_idx[move.promotion]
        # Promotion indices start after standard moves (4096)
        # There are 128 possible pawn promotions (64 white + 64 black), each with 4 promotions.
        promotion_from_to_index = None

        # White promotion: from rank 7 to rank 8
        if 48 <= from_sq <= 55 and 56 <= to_sq <= 63:
            promotion_from_to_index = (from_sq - 48) * 8 + (to_sq - 56)  # 0–63
        # Black promotion: from rank 2 to rank 1
        elif 8 <= from_sq <= 15 and 0 <= to_sq <= 7:
            promotion_from_to_index = 64 + (from_sq - 8) * 8 + to_sq  # 64–127
        else:
            raise ValueError("Invalid promotion move")

        return 4096 + promotion_from_to_index * 4 + promo_offset  # 4096–4607
    else:
        # Standard moves
        return from_sq * 64 + to_sq  # 0–4095

class ChessStockfishDataset(Dataset):
    def __init__(self, pgn_path, cache_path=None, max_games=10000, eval_depth=6):
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            self.data = torch.load(cache_path)
            return

        self.data = []
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        
        with open(pgn_path) as f:
            total = 0
            none_count = 0
            min_val = float('inf')
            max_val = float('-inf')
            
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            pgn_iterator = iter(lambda: chess.pgn.read_game(f), None)

            for count, game in tqdm(enumerate(pgn_iterator), total=max_games, desc="Games"):
                if count >= max_games:
                    break
                if game is None:
                    break

                board = game.board()
                for move in game.mainline_moves():
                    state = board_to_tensor(board)
                    move_index = move_to_index(move)

                    # Evaluate board before the move
                    info = engine.analyse(board, chess.engine.Limit(depth=eval_depth))
                    cp_score = info["score"].white().score(mate_score=10000) # Convert to pawn advantage
                    
                    total += 1
                    if cp_score is None:
                        none_count += 1
                        continue  # skip this position
    
                    min_val = min(min_val, cp_score)
                    max_val = max(max_val, cp_score)
                    
                    eval_scaled = scale_cp(cp_score)

                    self.data.append((state, move_index, eval_scaled))
                    board.push(move)

            engine.quit()

        
        print(f"Total samples: {total}, None values: {none_count}, Min: {min_val}, Max: {max_val}")
        if cache_path:
            print(f"Saving dataset to {cache_path}")
            torch.save(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, move_idx, value = self.data[idx]
        return state, torch.tensor(move_idx), torch.tensor(value, dtype=torch.float32)
    
    
if __name__ == "__main__":
    # Example usage
    dataset = ChessStockfishDataset("src/data/Carlsen.pgn", cache_path="src/data/chess_dataset_normalized.pt")
    print(f"Loaded {len(dataset)} samples.")
    
    # Example of accessing a sample
    sample_state, sample_move, sample_value = dataset[0]
    print(f"Sample state shape: {sample_state.shape}, move index: {sample_move}, value: {sample_value}")
    
    # stop the script from running
    exit(0)