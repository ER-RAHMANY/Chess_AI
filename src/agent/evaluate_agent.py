import chess
import chess.engine

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent.mcts_agent import MCTSAgent
from src.agent.model import ChessNet

from tqdm import tqdm



stockfish_path = "src/fairy-stockfish_x86-64.exe"

model = ChessNet()
model.load_state_dict(torch.load("best_model.pth"))  # Load your trained model

agent = MCTSAgent(model, iterations=500)  # adjust iterations as needed

stockfish_elo = 500

def play_game(agent, stockfish_elo=1000, play_as_white=True):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({
    'UCI_LimitStrength': True,
    'UCI_Elo': stockfish_elo  # or 400, 800, etc.
})

    while not board.is_game_over():
        if board.turn == chess.WHITE and play_as_white or board.turn == chess.BLACK and not play_as_white:
            move = agent.select_move(board.fen())
            board.push(chess.Move.from_uci(move))
        else:
            result = engine.play(board, chess.engine.Limit(depth=12))
            board.push(result.move)

    result = board.result()  # '1-0', '0-1', '1/2-1/2'
    engine.quit()
    return result

# Example match loop
results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
num_games = 10

agent_results = []

print("Starting evaluation...")
for i in tqdm(range(num_games), desc="Games", unit="game"):
    play_white = i % 2 == 0
    res = play_game(agent, stockfish_elo, play_as_white=play_white)
    results[res] += 1

    # Convert to agent's result
    if res == '1-0':
        agent_results.append(1 if play_white else 0)
    elif res == '0-1':
        agent_results.append(0 if play_white else 1)
    else:
        agent_results.append(0.5)
    print(f"Game {i+1}: {'Bot as White' if play_white else 'Bot as Black'} | Result: {res}")

# Compute win rate
# wins = results['1-0'] if play_white else results['0-1']
# losses = results['0-1'] if play_white else results['1-0']
# draws = results['1/2-1/2']
total = num_games
score = sum(agent_results) / total

# Calculate Elo rating based on the score
import math
if score == 1.0:
    print("Bot won all games — Elo is significantly higher than Stockfish's estimated level.")
elif score == 0.0:
    print("Bot lost all games — Elo is significantly lower than Stockfish's estimated level.")
else:
    estimated_elo = stockfish_elo + 400 * math.log10(score / (1 - score))
    print(f"Estimated Elo: {estimated_elo:.2f}")