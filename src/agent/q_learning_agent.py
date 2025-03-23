import random
import chess
from collections import defaultdict
from .random_agent import BaseAgent  # Assuming BaseAgent is defined in random_agent.py

class QLearningAgent(BaseAgent):
    def __init__(self, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initializes the Q-learning agent with the given hyperparameters.
        The reward structure in the environment is based on capturing black pieces:
            - Pawn capture: +1
            - Knight capture: +3
            - Bishop capture: +3
            - Rook capture: +5
            - Queen capture: +9
        Args:
            learning_rate (float): The step size (alpha).
            discount_factor (float): The discount factor (gamma).
            exploration_rate (float): Initial probability of taking a random move.
            exploration_decay (float): Decay factor for epsilon per episode.
            min_exploration_rate (float): Lower bound for epsilon.
        """
        super().__init__()
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate

        # Q-table mapping: {state (FEN): {action (UCI): Q-value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # To record the evolution of cumulative rewards over episodes
        self.reward_history = []

    def select_move(self, state):
        """
        Selects a move using an epsilon-greedy strategy.
        Args:
            state (str): Board state in FEN notation.
        Returns:
            move (str): Chosen legal move in UCI format.
        """
        board = chess.Board(state)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # With probability epsilon, choose a random move (exploration)
        if random.random() < self.epsilon:
            return random.choice(legal_moves).uci()
        else:
            # Otherwise, choose the move with the highest Q-value
            state_actions = self.q_table[state]
            best_move = None
            best_value = float("-inf")
            for move in legal_moves:
                move_uci = move.uci()
                q_value = state_actions.get(move_uci, 0.0)
                if q_value > best_value:
                    best_value = q_value
                    best_move = move_uci
            return best_move if best_move is not None else random.choice(legal_moves).uci()

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Q-learning update rule.
        The reward here is immediate and is based on capturing black pieces.
        Args:
            state (str): The previous state (FEN).
            action (str): The action taken (UCI format).
            reward (float): Immediate reward received (capture-based).
            next_state (str): The new state (FEN).
            done (bool): Whether the episode ended.
        """
        current_q = self.q_table[state][action]
        board_next = chess.Board(next_state)
        legal_moves = list(board_next.legal_moves)

        if done or not legal_moves:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state].get(move.uci(), 0.0) for move in legal_moves)
        
        # Q-learning update rule:
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max Q(s',a') - Q(s,a))
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def decay_exploration(self):
        """
        Decays the exploration rate epsilon, ensuring it does not fall below a minimum threshold.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def record_episode_reward(self, episode_reward):
        """
        Records the cumulative reward for an episode.
        Args:
            episode_reward (float): The total reward obtained in an episode.
        """
        self.reward_history.append(episode_reward)

    def get_reward_history(self):
        """
        Returns the recorded cumulative rewards per episode.
        """
        return self.reward_history
