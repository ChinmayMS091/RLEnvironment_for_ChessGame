
import sys
import os

# Add the current directory to sys.path so we can import the environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from server.reinforcement_learning_env_environment import ReinforcementLearningEnvironment
    from models import ReinforcementLearningAction
    import chess
    
    env = ReinforcementLearningEnvironment()
    obs = env.reset()
    print(f"Initial FEN: {obs.board_fen}")
    print(f"Legal moves: {obs.legal_moves[:5]}... (Total: {len(obs.legal_moves)})")
    
    # Try a simple move
    move = "e2e4"
    if move in obs.legal_moves:
        action = ReinforcementLearningAction(move_uci=move)
        obs = env.step(action)
        print(f"Move {move} played.")
        print(f"New FEN: {obs.board_fen}")
        print(f"Reward: {obs.reward}")
        print(f"Done: {obs.done}")
    else:
        print(f"Move {move} is not legal!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
