# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reinforcement Learning chess environment implementation.
"""

from uuid import uuid4

import chess
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ReinforcementLearningAction, ReinforcementLearningObservation
    from ..tasks import TASKS
except (ImportError, ModuleNotFoundError):
    try:
        from models import ReinforcementLearningAction, ReinforcementLearningObservation
        from tasks import TASKS
    except ImportError:
        # Fallback for when running from a different parent directory
        from reinforcement_learning_env.models import ReinforcementLearningAction, ReinforcementLearningObservation
        from reinforcement_learning_env.tasks import TASKS


class ReinforcementLearningEnvironment(Environment):
    """
    A single-agent chess environment for reinforcement learning.
    
    Includes 3 standard tasks (Easy, Medium, Hard) and a trajectory-based
    reward function.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MATERIAL_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    def __init__(self):
        """Initialize the chess environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._board = chess.Board()
        self._done = False
        self._last_move_uci: str | None = None
        self._terminal_result: str | None = None
        self._terminal_outcome: str | None = None
        self._terminal_reason: str | None = None
        self._current_task_idx = 0
        self._last_material_score = 0
        self._initial_material_score = 0

    @property
    def current_task(self):
        return TASKS[self._current_task_idx]

    def _get_material_score(self, board: chess.Board) -> int:
        score = 0
        for pt, val in self.MATERIAL_VALUES.items():
            score += len(board.pieces(pt, chess.WHITE)) * val
            score -= len(board.pieces(pt, chess.BLACK)) * val
        return score

    def _compute_reward(self, board: chess.Board, move: chess.Move, is_legal: bool) -> float:
        # 1. PENALTY FOR ILLEGAL MOVES (AS REQUESTED)
        if not is_legal:
            return -1.0 # Heavy negative point for making a mistake
            
        # 2. REWARD FOR LEGAL MOVES (Success)
        reward = 0.05 # Small base reward for being legal
        
        # Win/Loss rewards
        if board.is_checkmate():
            return 10.0 # Huge success!
        
        # Piece capture rewards (Material Advantage)
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        # If we just took a piece
        if board.is_capture(move):
            target_piece = board.piece_at(move.to_square)
            if target_piece:
                reward += values.get(target_piece.piece_type, 0)
                
        # Mobility & Center control bonus
        # (Encourages model to not just sit still)
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            reward += 0.05
            
        return reward

    def _calculate_reward(self, done: bool, acting_side: str) -> float:
        # 1. Base Terminal Reward
        if done:
            if self._terminal_outcome == f"{acting_side}_win":
                return 1.0
            elif self._terminal_outcome == "draw":
                return 0.0
            else:
                return -1.0

        # 2. Trajectory Signal: Material Change
        current_material = self._get_material_score(self._board)
        # We want to maximize our own color's material
        delta = current_material - self._last_material_score
        # Normalize delta slightly; capture a pawn (100) -> 0.1 reward
        material_reward = delta / 1000.0 if acting_side == "white" else -delta / 1000.0
        
        # 3. Positional Signal: Center Control
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        center_control = sum(1 for sq in center_squares if self._board.piece_at(sq) and 
                              ("white" if self._board.color_at(sq) == chess.WHITE else "black") == acting_side)
        position_reward = center_control * 0.01

        # 4. Mobility Signal
        mobility_reward = len(list(self._board.legal_moves)) * 0.001

        self._last_material_score = current_material
        return material_reward + position_reward + mobility_reward

    @staticmethod
    def _color_name(color: bool) -> str:
        return "white" if color == chess.WHITE else "black"

    def _set_terminal_loss_for_current_player(self) -> None:
        if self._board.turn == chess.WHITE:
            self._terminal_result = "0-1"
            self._terminal_outcome = "black_win"
        else:
            self._terminal_result = "1-0"
            self._terminal_outcome = "white_win"

    def _sync_terminal_status_from_board(self) -> None:
        outcome = self._board.outcome(claim_draw=True)
        if outcome is None:
            return

        self._terminal_result = outcome.result()
        if outcome.winner is None:
            self._terminal_outcome = "draw"
        elif outcome.winner == chess.WHITE:
            self._terminal_outcome = "white_win"
        else:
            self._terminal_outcome = "black_win"
        self._terminal_reason = outcome.termination.name.lower()

    def _build_observation(
        self, reward: float, done: bool, metadata: dict[str, str | int] | None = None
    ) -> ReinforcementLearningObservation:
        if self._terminal_result is None or self._terminal_outcome is None:
            game_result = "*"
            outcome = "ongoing"
        else:
            game_result = self._terminal_result
            outcome = self._terminal_outcome

        # Add grading score to metadata if done
        metadata = metadata or {}
        if done:
            metadata["task_name"] = self.current_task.name

        return ReinforcementLearningObservation(
            board_fen=self._board.fen(),
            legal_moves=[move.uci() for move in self._board.legal_moves],
            side_to_move=self._color_name(self._board.turn),
            last_move_uci=self._last_move_uci,
            is_check=self._board.is_check(),
            game_result=game_result,
            outcome=outcome,
            done=done,
            reward=reward,
            metadata=metadata,
        )

    def reset(self, task_idx: int = 0) -> ReinforcementLearningObservation:
        """
        Reset to a specific task.
        """
        self._current_task_idx = max(0, min(task_idx, len(TASKS) - 1))
        task = self.current_task
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._board = chess.Board(task.fen)
        self._done = False
        self._last_move_uci = None
        self._terminal_result = None
        self._terminal_outcome = None
        self._terminal_reason = None
        
        self._last_material_score = self._get_material_score(self._board)
        self._initial_material_score = self._last_material_score

        return self._build_observation(
            reward=0.0,
            done=False,
            metadata={
                "message": f"Reset to task: {task.name} ({task.difficulty})",
                "goal": task.goal_description,
                "step": 0
            },
        )

    def step(self, action: ReinforcementLearningAction) -> ReinforcementLearningObservation:  # type: ignore[override]
        if self._done:
            return self._build_observation(
                reward=0.0,
                done=True,
                metadata={
                    "error": "Episode is terminated.",
                    "step": self._state.step_count,
                },
            )

        self._state.step_count += 1
        acting_side = self._color_name(self._board.turn)

        if action.resign:
            self._done = True
            self._terminal_reason = f"{acting_side}_resigned"
            self._set_terminal_loss_for_current_player()
            return self._build_observation(
                reward=-1.0,
                done=True,
                metadata={
                    "player": acting_side,
                    "step": self._state.step_count,
                    "terminal_reason": self._terminal_reason,
                },
            )

        move_uci = action.move_uci.strip().lower()
        if not move_uci:
            # Illegal/Incomplete behavior: negative reward
            self._done = True
            self._terminal_reason = "missing_move"
            self._set_terminal_loss_for_current_player()
            return self._build_observation(reward=-1.0, done=True)

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            self._done = True
            self._terminal_reason = "invalid_uci"
            self._set_terminal_loss_for_current_player()
            return self._build_observation(reward=-1.0, done=True)

        if move not in self._board.legal_moves:
            self._done = True
            self._terminal_reason = "illegal_move"
            self._set_terminal_loss_for_current_player()
            return self._build_observation(reward=-1.0, done=True)

        self._board.push(move)
        self._last_move_uci = move.uci()

        if self._board.is_game_over(claim_draw=True):
            self._done = True
            self._sync_terminal_status_from_board()

        reward = self._calculate_reward(self._done, acting_side)
        
        metadata: dict[str, str | int] = {
            "player": acting_side,
            "step": self._state.step_count,
            "applied_move": self._last_move_uci,
        }

        return self._build_observation(reward=reward, done=self._done, metadata=metadata)

    @property
    def state(self) -> State:
        return self._state
