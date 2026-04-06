
import os
import sys
import chess
from dotenv import load_dotenv
from openai import OpenAI
from reinforcement_learning_env.client import ReinforcementLearningEnv
from reinforcement_learning_env.models import ReinforcementLearningAction

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt_move(board_fen, legal_moves):
    """Ask GPT-4o for the best move in UCI format."""
    prompt = f"""
Current Chess Board (FEN): {board_fen}
Legal Moves (UCI): {', '.join(legal_moves)}

You are a chess master. Pick the absolute best move from the legal moves list.
Respond ONLY with the UCI move string (e.g. 'e2e4').
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        move = response.choices[0].message.content.strip().lower()
        # Basic validation
        if move in legal_moves:
            return move
        # Fallback to random if GPT fails or hallucinates
        import random
        return random.choice(legal_moves)
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        import random
        return random.choice(legal_moves)

def run_baseline():
    base_url = "http://localhost:8000"
    results = []

    try:
        with ReinforcementLearningEnv(base_url=base_url).sync() as env:
            for task_idx in range(3):
                # Reset to specific task
                # We need to ensure the server's reset() accepts task_idx
                # (I updated it to accept it)
                step_result = env.reset(task_idx=task_idx)
                obs = step_result.observation
                print(f"\n--- Starting Task {task_idx+1}: {obs.metadata.get('task_name')} ---")
                
                step_count = 0
                total_reward = 0
                while not step_result.done and step_count < 10:
                    move_uci = get_gpt_move(obs.board_fen, obs.legal_moves)
                    print(f"Step {step_count+1}: GPT played {move_uci}")
                    
                    step_result = env.step(ReinforcementLearningAction(move_uci=move_uci))
                    obs = step_result.observation
                    total_reward += (step_result.reward or 0)
                    step_count += 1
                
                final_score = obs.metadata.get("task_score", 0.0)
                results.append({
                    "task": obs.metadata.get("task_name"),
                    "score": final_score,
                    "reward": total_reward
                })
                print(f"Task Finished. Score: {final_score}")

    except Exception as e:
        print(f"Error: {e}. Is the server running at {base_url}?")
        return

    print("\n" + "="*30)
    print("BASELINE PERFORMANCE REPORT")
    print("="*30)
    for res in results:
        print(f"{res['task']}: Score {res['score']:.2f}, Total Reward {res['reward']:.2f}")

if __name__ == "__main__":
    run_baseline()
