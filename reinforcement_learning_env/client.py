# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reinforcement Learning chess environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ReinforcementLearningAction, ReinforcementLearningObservation


class ReinforcementLearningEnv(
    EnvClient[ReinforcementLearningAction, ReinforcementLearningObservation, State]
):
    """
    Client for the Reinforcement Learning chess environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ReinforcementLearningEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.board_fen)
        ...
        ...     result = client.step(ReinforcementLearningAction(move_uci="e2e4"))
        ...     print(result.observation.last_move_uci)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ReinforcementLearningEnv.from_docker_image("reinforcement_learning_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ReinforcementLearningAction(move_uci="e2e4"))
        ... finally:
        ...     client.close()
    """

    def _reset_payload(self, **kwargs) -> Dict:
        """Pass reset parameters (like task_idx) to the server."""
        return kwargs

    def _step_payload(self, action: ReinforcementLearningAction) -> Dict:
        """
        Convert ReinforcementLearningAction to JSON payload for step message.

        Args:
            action: ReinforcementLearningAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "move_uci": action.move_uci,
            "resign": action.resign,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ReinforcementLearningObservation]:
        """
        Parse server response into StepResult[ReinforcementLearningObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ReinforcementLearningObservation
        """
        obs_data = payload.get("observation", {})
        observation = ReinforcementLearningObservation(
            board_fen=obs_data.get("board_fen", ""),
            legal_moves=obs_data.get("legal_moves", []),
            side_to_move=obs_data.get("side_to_move", "white"),
            last_move_uci=obs_data.get("last_move_uci"),
            is_check=obs_data.get("is_check", False),
            game_result=obs_data.get("game_result", "*"),
            outcome=obs_data.get("outcome", "ongoing"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
