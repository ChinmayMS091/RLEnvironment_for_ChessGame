# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Reinforcement Learning Env chess environment.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ReinforcementLearningAction(Action):
    """Action for the chess environment."""

    move_uci: str = Field(
        default="",
        description="Move in UCI format (for example: 'e2e4', 'g1f3', 'e7e8q').",
    )
    resign: bool = Field(default=False, description="Whether to resign the game.")


class ReinforcementLearningObservation(Observation):
    """Observation describing the current chess position and episode status."""

    board_fen: str = Field(default="", description="Current board position as FEN.")
    legal_moves: list[str] = Field(
        default_factory=list, description="Legal moves available in UCI format."
    )
    side_to_move: Literal["white", "black"] = Field(
        default="white", description="Side to move."
    )
    last_move_uci: str | None = Field(
        default=None, description="Most recent move played in UCI format."
    )
    is_check: bool = Field(default=False, description="Whether side to move is in check.")
    game_result: str = Field(
        default="*",
        description="Game result in PGN style ('1-0', '0-1', '1/2-1/2', or '*').",
    )
    outcome: Literal["ongoing", "white_win", "black_win", "draw"] = Field(
        default="ongoing", description="High-level outcome label."
    )
