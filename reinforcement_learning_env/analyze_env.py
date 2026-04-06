
import sys
import os
import chess
import random

# Ensure local imports work by adding the current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from server.reinforcement_learning_env_environment import ReinforcementLearningEnvironment
    from models import ReinforcementLearningAction
except ImportError:
    # If paths are slightly different during execution
    from reinforcement_learning_env.server.reinforcement_learning_env_environment import ReinforcementLearningEnvironment
    from reinforcement_learning_env.models import ReinforcementLearningAction

def analyze():
    # 1. Initialize the environment
    env = ReinforcementLearningEnvironment()
    obs = env.reset()
    
    print("\n" + "="*40)
    print("CHESS RL ENVIRONMENT ANALYSIS")
    print("="*40)
    
    board = chess.Board(obs.board_fen)
    print("\nInitial State:")
    print(board)
    print(f"Side to move: {obs.side_to_move}")

    step_count = 0
    # Run a short simulation of 10 random moves
    while not obs.done and step_count < 10:
        step_count += 1
        
        # Pick a random legal move
        if not obs.legal_moves:
            print("No legal moves available!")
            break
            
        move_uci = random.choice(obs.legal_moves)
        print(f"\nStep {step_count}: Playing {move_uci}")
        
        action = ReinforcementLearningAction(move_uci=move_uci)
        obs = env.step(action)
        
        # Update and show board
        board = chess.Board(obs.board_fen)
        print(board)
        
        # Analyze metrics
        print(f"Reward: {obs.reward}")
        print(f"Status: {obs.outcome}")
        if obs.is_check:
            print("CHECK!")
            
    if obs.done:
        print("\n" + "="*40)
        print("EPISODE TERMINATED")
        print(f"Final Outcome: {obs.outcome}")
        print(f"Game Result: {obs.game_result}")
        print("="*40)
    else:
        print("\nAnalysis paused after 10 moves.")

if __name__ == "__main__":
    analyze()
