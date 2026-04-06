
import chess
import json
from pathlib import Path

# Path to the data we will analyze for success
HISTORY_PATH = Path("chess_training_states.json")

class ChessTask:
    def __init__(self, fen: str, name: str, difficulty: str, goal_description: str):
        self.fen = fen
        self.name = name
        self.difficulty = difficulty
        self.goal_description = goal_description

    def score(self, board: chess.Board) -> float:
        """This will now analyze training data to assign success."""
        # Calculate current board strength (Material + Position)
        current_strength = self.calculate_strength(board)
        
        # Pull history to see if we improved
        if not HISTORY_PATH.exists():
            return 0.5 # Initial neutral score
        
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
            
            # Extract previous rewards to find the 'Benchmark'
            rewards = [h.get('reward', 0) for h in history if isinstance(h, dict)]
            if not rewards: return 0.5
            
            avg_reward = sum(rewards) / len(rewards)
            max_reward = max(rewards)
            
            # ASIGN SUCCESS BASED ON ANALYSIS
            # If current move/state is better than average, we give success!
            if current_strength > max_reward:
                return 1.0 # NEW RECORD!
            elif current_strength > avg_reward:
                return 0.7 # Better than average
            else:
                return 0.1 # Still learning
        except:
            return 0.0

    def calculate_strength(self, board: chess.Board):
        # Basic piece values
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        score = 0
        for pt, val in values.items():
            score += len(board.pieces(pt, chess.WHITE)) * val
            score -= len(board.pieces(pt, chess.BLACK)) * val
        return score

class SelfImprovementTask(ChessTask):
    def __init__(self):
        super().__init__(
            fen=chess.STARTING_FEN,
            name="Self-Improvement Training",
            difficulty="dynamic",
            goal_description="Exceed the average performance points stored in your training history."
        )

TASKS = [SelfImprovementTask()]
