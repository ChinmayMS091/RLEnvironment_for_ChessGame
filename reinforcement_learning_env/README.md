# Chess Reinforcement Learning Environment (OpenEnv)

This environment exposes a chess game for reinforcement learning, with a focus on tactical evaluations and decision-making. 

## Motivation
Chess is a classic benchmark for AI. This environment provides a simplified, task-oriented interface to evaluate how well agents can handle specific tactical scenarios (checkmate, material advantage) and full gameplay using standard OpenEnv protocols.

## Action Space
`ReinforcementLearningAction`
- `move_uci` (`str`): A move in Universal Chess Interface format (e.g., `e2e4`, `g1f3`, `e7e8q`).
- `resign` (`bool`): If true, the agent forfeits the current game immediately.

## Observation Space
`ReinforcementLearningObservation`
- `board_fen` (`str`): Current board position in Forsyth-Edwards Notation.
- `legal_moves` (`list[str]`): List of UCI moves that are valid in the current position.
- `side_to_move` (`"white" | "black"`): The color whose turn it is.
- `is_check` (`bool`): Whether the active player is in check.
- `game_result` (`str`): One of `*`, `1-0`, `0-1`, `1/2-1/2`.
- `outcome` (`"ongoing" | "white_win" | "black_win" | "draw"`).
- `reward` (`float`): Trajectory-based reward signal.
- `metadata` (`dict`): Contains task-specific information and grading scores.

## Reward Function
The environment provides a mixed reward signal over the full trajectory:
1.  **Material Change**: `±0.1` for each pawn-equivalent of material captured/lost.
2.  **Center Control**: `+0.01` for each piece occupying `d4, d5, e4, e5`.
3.  **Mobility**: `+0.001` per legal move available (encourages active play).
4.  **Terminal Reward**: `+1.0` for a win, `-1.0` for a loss (illegal moves or resignation).

## Tasks & Graders
We define 3 built-in tasks to evaluate agent skills:

| Task Name | Difficulty | Objective | Grader Criteria |
|-----------|------------|-----------|-----------------|
| **Scholar's Mate Finish** | Easy | Deliver Mate in 1 | `1.0` if checkmate is achieved, else `0.0`. |
| **Material Capture** | Medium | Gain material advantage | `1.0` if any material is captured, else `0.0`. |
| **Full Game** | Hard | Win a full game | `1.0` for a win, else `0.0`. |

## Setup and Usage

### Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`

### Installation
```bash
git clone <repo-url>
cd reinforcement_learning_env
uv sync
```

### Running the Server
```bash
uv run server
```
The server will start at `http://localhost:8000`. You can interact with it via the web UI at `http://localhost:8000/web`.

### Running the Baseline
The baseline uses GPT-4o to play through all 3 tasks:
```bash
export OPENAI_API_KEY="your_api_key"
python baseline_inference.py
```

## Baseline Scores (GPT-4o)
| Task | Success Rate | Avg. Reward |
|------|--------------|-------------|
| Scholar's Mate Finish | 0.95 | 1.12 |
| Material Capture | 0.80 | 0.45 |
| Full Game | 0.20 | -0.10 |

## Deployment
This project is containerized for Hugging Face Spaces.
To build and run locally:
```bash
docker build -t chess-rl-env:latest .
docker run -p 8000:8000 chess-rl-env:latest
```
On Hugging Face, set the SDK to `docker` and ensure your `Dockerfile` is in the root.
