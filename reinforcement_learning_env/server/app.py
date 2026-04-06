# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Reinforcement Learning Env Environment.

This module creates an HTTP server that exposes the ReinforcementLearningEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ReinforcementLearningAction, ReinforcementLearningObservation
    from .reinforcement_learning_env_environment import ReinforcementLearningEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from models import ReinforcementLearningAction, ReinforcementLearningObservation
        from reinforcement_learning_env_environment import ReinforcementLearningEnvironment
    except ImportError:
        # Fallback for when running from a different parent directory
        from reinforcement_learning_env.models import ReinforcementLearningAction, ReinforcementLearningObservation
        from reinforcement_learning_env.server.reinforcement_learning_env_environment import ReinforcementLearningEnvironment


# Create the app with web interface and README integration
app = create_app(
    ReinforcementLearningEnvironment,
    ReinforcementLearningAction,
    ReinforcementLearningObservation,
    env_name="reinforcement_learning_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m reinforcement_learning_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn reinforcement_learning_env.server.app:app --workers 4
    """
    import argparse
    import sys
    import uvicorn

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=port)
        args, _ = parser.parse_known_args()
        port = args.port

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
