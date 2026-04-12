
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
        """
        Evaluate the agent's performance based on:
        1. Material advantage at end of episode
        2. Improvement over historical training runs
        3. Game outcome (checkmate, draw, etc.)
        """
        # Start with material-based score
        material_score = self.calculate_strength(board)

        # Check for game-ending conditions
        if board.is_checkmate():
            # Who won?
            if board.turn == chess.BLACK:  # White delivered checkmate
                return 0.99
            else:
                return 0.05  # We got checkmated

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.40  # Draw is okay but not great

        # Pull history to measure improvement
        if not HISTORY_PATH.exists():
            return max(0.10, min(0.90, material_score))

        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)

            # Calculate average total reward per episode (group by ~30 steps)
            rewards = [h.get('reward', 0) for h in history if isinstance(h, dict)]
            if not rewards:
                return max(0.10, min(0.90, material_score))

            # Use the total cumulative reward as the benchmark
            total_reward = sum(rewards)
            num_episodes = max(1, len(rewards) // 30)  # ~30 steps per episode
            avg_episode_reward = total_reward / num_episodes

            # Score based on material advantage + learning trajectory
            # Material score: normalize to 0-1 range (max theoretical advantage ~39)
            normalized_material = max(0.0, min(1.0, (material_score + 20) / 40.0))

            # Learning bonus: reward if we're improving
            learning_bonus = 0.0
            if len(rewards) > 60:  # At least 2 episodes of data
                mid = len(rewards) // 2
                early_avg = sum(rewards[:mid]) / mid
                recent_avg = sum(rewards[mid:]) / (len(rewards) - mid)
                if recent_avg > early_avg:
                    learning_bonus = 0.15  # Agent is improving!

            final_score = min(0.99, max(0.01, normalized_material * 0.7 + learning_bonus + 0.15))
            return round(final_score, 4)

        except Exception:
            return max(0.10, min(0.90, material_score))

    def calculate_strength(self, board: chess.Board) -> float:
        """Calculate normalized material advantage (0.0 to 1.0 scale)."""
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        raw_score = 0
        for pt, val in values.items():
            raw_score += len(board.pieces(pt, chess.WHITE)) * val
            raw_score -= len(board.pieces(pt, chess.BLACK)) * val
        # Normalize: raw_score ranges roughly -39 to +39, map to 0.0-1.0
        return max(0.0, min(1.0, (raw_score + 20) / 40.0))

class SelfImprovementTask(ChessTask):
    def __init__(self):
        super().__init__(
            fen=chess.STARTING_FEN,
            name="Self-Improvement Training",
            difficulty="dynamic",
            goal_description="Exceed the average performance points stored in your training history."
        )

class EndgameTask(ChessTask):
    def __init__(self):
        super().__init__(
            fen="8/p4pkp/1p4p1/2r5/8/1P3RP1/P4P1P/6K1 w - - 0 1",
            name="Rook and Pawn Endgame",
            difficulty="hard",
            goal_description="Navigate a realistic rook and pawn endgame to gain an advantage."
        )

class OpeningTask(ChessTask):
    def __init__(self):
        super().__init__(
            fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            name="Italian Game Opening",
            difficulty="easy",
            goal_description="Continue strongly from the Italian Game."
        )

class MidgameTacticalTask(ChessTask):
    def __init__(self):
        super().__init__(
            fen="r1bq1rk1/ppp2ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQ - 4 6",
            name="Midgame Tactics",
            difficulty="medium",
            goal_description="Gain a tactical advantage in the midgame."
        )

TASKS = [SelfImprovementTask(), OpeningTask(), EndgameTask(), MidgameTacticalTask()]
