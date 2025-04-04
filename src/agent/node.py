import chess
from chess import Move
import math
from src.agent import config

class Edge:
    def __init__(self, input_node: "Node", output_node: "Node", action: Move, prior: float):
        self.input_node = input_node
        self.output_node = output_node
        self.action = action

        self.player_turn = self.input_node.state.split(" ")[1] == "w"

        # each action stores 4 numbers:
        self.N = 0  # amount of times this action has been taken (=visit count)
        self.W = 0  # total action-value
        self.P = prior  # prior probability of selecting this action

    def __eq__(self, edge: object) -> bool:
        if isinstance(edge, Edge):
            return self.action == edge.action and self.input_node.state == edge.input_node.state
        else:
            return NotImplemented

    def __str__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.upper_confidence_bound()}"

    def __repr__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.upper_confidence_bound()}"

    def upper_confidence_bound(self, noise: float) -> float:
        exploration_rate = math.log((1 + self.input_node.N + config.C_base) / config.C_base) + config.C_init
        ucb = exploration_rate * (self.P * noise) * (math.sqrt(self.input_node.N) / (1 + self.N))
        if self.input_node.turn == chess.WHITE:
            return self.W / (self.N + 1) + ucb 
        else:
            return -(self.W / (self.N + 1)) + ucb

class Node:
    def __init__(self, state: str):
        """
        A node is a state inside the MCTS tree.
        """
        self.state = state
        self.turn = chess.Board(state).turn
        # the edges connected to this node
        self.edges: list[Edge] = []
        # the visit count for this node
        self.N = 0

        self.value = 0

    def __eq__(self, node: object) -> bool:
        """
        Check if two nodes are equal.
        Two nodes are equal if the state is the same
        """
        if isinstance(node, Node):
            return self.state == node.state
        else:
            return NotImplemented

    def step(self, action: Move) -> str:
        """
        Take a step in the game, returns new state
        """
        board = chess.Board(self.state)
        board.push(action)
        new_state = board.fen()
        del board
        return new_state

    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        """
        board = chess.Board(self.state)
        return board.is_game_over()

    def is_leaf(self) -> bool:
        """
        Check if the current node is a leaf node.
        """
        return self.N == 0

    def add_child(self, child, action: Move, prior: float) -> Edge:
        """
        Add a child node to the current node.

        Returns the created edge between the nodes
        """
        edge = Edge(input_node=self, output_node=child, action=action, prior=prior)
        self.edges.append(edge)
        return edge

    def get_all_children(self):
        """
        Get all children of the current node and their children, recursively
        """
        children = []
        for edge in self.edges:
            children.append(edge.output_node)
            children.extend(edge.output_node.get_all_children())
        return children

    def get_edge(self, action) -> Edge:
        """
        Get the edge between the current node and the child node with the given action.
        """
        for edge in self.edges:
            if edge.action == action:
                return edge
        return None