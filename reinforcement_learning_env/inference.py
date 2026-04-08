"""
MANDATORY HACKATHON INFERENCE SCRIPT
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""
import os
import json
import random
import chess
import warnings
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, logging as transformer_logging
from reinforcement_learning_env.client import ReinforcementLearningEnv
from reinforcement_learning_env.models import ReinforcementLearningAction

# [MANUAL OVERRIDE] Set to True to see the boards. Set to False for Hackathon Submission.
DEBUG = False

# 1. MANDATORY ENVIRONMENT CONFIGURATION
load_dotenv()
warnings.filterwarnings("ignore")
transformer_logging.set_verbosity_error()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
HF_TOKEN = os.getenv("HF_TOKEN", "not-set")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "chess-rl-env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", HF_TOKEN)

# Initialize Engines
client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    hf_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
except Exception:
    hf_generator = None

# 2. PERSISTENT LEARNING (History Tracking)
LEARNING_FILE = Path("chess_training_states.json")

def get_wisdom(fen, legal_moves):
    if not LEARNING_FILE.exists(): return None
    try:
        with open(LEARNING_FILE, "r") as f:
            history = json.load(f)
        match = [h for h in history if h['fen'] == fen]
        if match:
            best = max(match, key=lambda x: x['reward'])
            if int(best['reward'] * 100) > 2: # Only pick if reward is significant
                if best['move'] in legal_moves: return best['move']
    except: pass
    return None

def update_learning(fen, move, reward):
    data = []
    if LEARNING_FILE.exists():
        try:
            with open(LEARNING_FILE, "r") as f:
                data = json.load(f)
        except: pass
    data.append({"fen": fen, "move": move, "reward": reward})
    with open(LEARNING_FILE, "w") as f:
        json.dump(data[-500:], f)

# 3. DUAL ENGINE (OpenAI -> HF Fallback)
def predict_move(board_fen, legal_moves):
    wisdom = get_wisdom(board_fen, legal_moves)
    prompt = f"FEN: {board_fen}. Memory: {wisdom}. Best Move UCI:"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
            max_tokens=10, temperature=0
        )
        text = response.choices[0].message.content.strip().lower()
        for m in legal_moves:
            if m in text: return m, None
    except Exception:
        if hf_generator:
            try:
                output = hf_generator(prompt, max_new_tokens=5, pad_token_id=50256, truncation=True)
                text = output[0]['generated_text'].replace(prompt, "").strip().lower()
                for m in legal_moves:
                    if m in text: return m, None
            except: pass
    return random.choice(legal_moves), "Fallback"

# 4. MAIN TASK LOOP
def run_benchmark():
    import sys
    if "--reset" in sys.argv and LEARNING_FILE.exists():
        LEARNING_FILE.unlink()
        print("Memory Cleared.")

    try:
        with ReinforcementLearningEnv(base_url="http://localhost:7860").sync() as env:
            for task_idx in range(4):
                step_result = env.reset(task_idx=task_idx)
                obs = step_result.observation
                
                # [START] Compliance
                task_name = "Task-" + str(task_idx)
                if hasattr(obs, 'metadata') and obs.metadata and "message" in obs.metadata:
                    msg = obs.metadata["message"]
                    task_name = msg.replace("Reset to task: ", "").split(" (")[0].replace(" ", "_")
                
                print(f"[START] task={task_name} env=Chess-RL-v1 model={MODEL_NAME}")
                
                rewards_list, step_count = [], 0
                final_success = False
                
                while not step_result.done and step_count < 30:
                    step_count += 1
                    current_fen = obs.board_fen
                    action_str, _ = predict_move(current_fen, obs.legal_moves)
                    
                    step_result = env.step(ReinforcementLearningAction(move_uci=action_str))
                    obs = step_result.observation
                    
                    reward = float(step_result.reward or 0.0)
                    rewards_list.append(reward)
                    update_learning(current_fen, action_str, reward)
                    
                    # [STEP] Compliance
                    done_bool = "true" if step_result.done or step_count >= 30 else "false"
                    error_msg = obs.metadata.get("error", "null") if hasattr(obs, 'metadata') and obs.metadata and "error" in obs.metadata else "null"
                    print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_bool} error={error_msg}")
                    
                    if DEBUG:
                        print("-" * 20 + f"\n{chess.Board(obs.board_fen)}\n" + "-" * 20)
                
                final_success = (sum(rewards_list) > 0)
                
                # [END] Compliance (MANDATORY FIELDS: success, steps, score, rewards)
                final_score = float(obs.metadata.get("task_score", 0.01)) if hasattr(obs, 'metadata') and obs.metadata and "task_score" in obs.metadata else 0.01
                if final_score <= 0.0: final_score = 0.01
                if final_score >= 1.0: final_score = 0.99
                
                rew_str = ",".join([f"{r:.2f}" for r in rewards_list]) if rewards_list else "0.00"
                print(f"[END] success={'true' if final_success else 'false'} steps={step_count} score={final_score:.2f} rewards={rew_str}")
                
    except Exception as e:
        import traceback
        traceback.print_exc()

    if DEBUG:
        print("\n" + "="*30 + f"\nMATCH COMPLETED\nTotal Reward Sum: {sum(rewards_list):.2f}\n" + "="*30)

if __name__ == "__main__":
    run_benchmark()