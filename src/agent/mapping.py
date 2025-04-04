import torch
import chess

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

# ===================
# Move Index Encoding (AlphaZero-style: 73x8x8)
# ===================

# Queen-like directions: N, NE, E, SE, S, SW, W, NW
DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1),
              (0, -1), (-1, -1), (-1, 0), (-1, 1)]

# Knight offsets
KNIGHT_OFFSETS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]

# Promotion directions: forward, left-capture, right-capture
PROMO_DIRECTIONS = [(0, 1), (-1, 1), (1, 1)]  # For white; mirrored for black
PROMO_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP]

# Total 73 move types
MOVE_TYPES = []

# Add queen-like moves
for dx, dy in DIRECTIONS:
    for dist in range(1, 8):
        MOVE_TYPES.append((dx * dist, dy * dist, None))

# Add knight moves
for dx, dy in KNIGHT_OFFSETS:
    MOVE_TYPES.append((dx, dy, None))

# Add promotion moves (no underpromoting to knight in this setup)
for dx, dy in PROMO_DIRECTIONS:
    for promo in PROMO_PIECES:
        MOVE_TYPES.append((dx, dy, promo))

assert len(MOVE_TYPES) == 73

# ===================
# Move ↔ Index Mapping
# ===================

def move_to_index(move, color=chess.WHITE):
    """Converts a chess.Move into (plane, row, col) index."""
    from_sq = move.from_square
    to_sq = move.to_square
    row, col = divmod(from_sq, 8)
    dx = (to_sq % 8) - col
    dy = (to_sq // 8) - row

    if color == chess.BLACK:
        dx, dy = -dx, -dy  # Flip for black's perspective

    for plane, (mx, my, promo) in enumerate(MOVE_TYPES):
        if dx == mx and dy == my:
            if move.promotion and promo != move.promotion:
                continue
            return (plane, 7 - row, col)  # row is flipped for display

    raise ValueError("Move not found in predefined move types")

def index_to_move(plane, row, col, board):
    """Converts (plane, row, col) back to a chess.Move object."""
    # Flip row back
    from_row = 7 - row
    from_sq = chess.square(col, from_row)
    dx, dy, promo = MOVE_TYPES[plane]

    to_col = col + dx
    to_row = from_row + dy
    if not (0 <= to_col < 8 and 0 <= to_row < 8):
        return None

    to_sq = chess.square(to_col, to_row)
    move = chess.Move(from_sq, to_sq, promotion=promo)
    if move in board.legal_moves:
        return move
    return None
