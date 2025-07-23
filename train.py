import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from HighwayEnvMod import highway_env
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env # Can be removed if not used elsewhere
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os
import torch
import shutil
import glob

import wandb
from wandb.integration.sb3 import WandbCallback

import traceback
import random


def generate_funny_name():
    adjectives = [
        "fluffy", "sparkly", "dizzy", "bouncy", "wobbly", "snuggly",
        "sassy", "grumpy", "loopy", "sleepy", "zippy", "wiggly", "clumsy"
    ]
    animals = [
        "panda", "duck", "sloth", "turtle", "unicorn", "otter",
        "penguin", "llama", "koala", "platypus", "giraffe", "hedgehog", "blobfish"
    ]
    return f"{random.choice(adjectives)}-{random.choice(animals)}-{random.randint(1000, 9999)}"


# --- Configuration ---
# Define the environments to test
ENVIRONMENTS = [
    "highway-fast-v0",
    "intersection-v0",
    "racetrack-v0"
]

# Define the algorithms and their hyperparameter grids
# Note: For simplicity, a small grid is provided. For real tuning, expand these.
# Also, consider action space compatibility (DQN for discrete, PPO/SAC for continuous/discrete)
ALGORITHMS = {
    "DQN": {
        "model": DQN,
        "hyperparams": {
            "learning_rate": [1e-4, 5e-4],
            "buffer_size": [10000, 50000],
            "learning_starts": [1000, 2000],
            "batch_size": [32, 64],
            "gamma": [0.99, 0.95],
            "train_freq": [(1, "episode"), (4, "step")], # (frequency, unit)
            "target_update_interval": [1000, 5000],
            "exploration_fraction": [0.1, 0.2]
        },
        "action_space_type": "Discrete"
    },
    "PPO": {
        "model": PPO,
        "hyperparams": {
            "learning_rate": [3e-4, 1e-4],
            "n_steps": [2048, 1024],
            "batch_size": [64, 128],
            "n_epochs": [10, 20],
            "gamma": [0.99, 0.98],
            "gae_lambda": [0.95, 0.9],
            "clip_range": [0.2, 0.1]
        },
        "action_space_type": "Both" # PPO handles both discrete and continuous
    },
    "SAC": {
        "model": SAC,
        "hyperparams": {
            "learning_rate": [3e-4, 1e-4],
            "buffer_size": [100000, 500000],
            "learning_starts": [100, 1000],
            "batch_size": [256, 128],
            "gamma": [0.99, 0.95],
            "tau": [0.005, 0.01],
            "train_freq": [(1, "episode")], # SAC often uses (1, "episode")
            "gradient_steps": [1, -1] # -1 means as many as steps done in env during rollout
        },
        "action_space_type": "Continuous" # SAC is primarily for continuous, though extensions exist
    }
}

TOTAL_TIMESTEPS = int(5e4) # Total timesteps for each training run
EVAL_FREQ = 5000          # Evaluate every N timesteps
N_EVAL_EPISODES = 10      # Number of episodes for evaluation
LOG_DIR = "./rl_logs"     # Directory for TensorBoard logs and saved models

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

highway_env.register_highway_envs()

# --- Helper Functions ---
def get_action_space_type(env):
    """Determines if the environment has a discrete or continuous action space."""
    if isinstance(env.action_space, gym.spaces.Discrete):
        return "Discrete"
    elif isinstance(env.action_space, gym.spaces.Box):
        return "Continuous"
    else:
        raise ValueError("Unsupported action space type.")

# MODIFIED: Removed Monitor wrapping from here
def create_env(env_id, render_mode=None, **kwargs):
    """Helper to create the environment."""
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    return env

# The train_and_evaluate function is not used in your main loop's structure.
# Therefore, WandB integration will be done directly in the main loop.
def train_and_evaluate(algorithm_name, env_id, model_class, hyperparams, total_timesteps, log_dir):
    """Trains and evaluates a single RL algorithm with a given set of hyperparameters."""
    print(f"\n--- Training {algorithm_name} on {env_id} with hyperparameters: {hyperparams} ---")

    # Create training environment
    train_env = create_env(env_id)

    # Determine policy based on action space (MlpPolicy is common for most observations)
    policy_type = "MlpPolicy"

    model = algo_config["model"](
        policy_type,
        current_train_env,
        verbose=0,
        tensorboard_log=run_log_dir,
        device="cuda",  # <<< force GPU usage if available
        **hparams
    )

    # Create evaluation environment
    eval_env = create_env(env_id)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, algorithm_name, env_id, "best_model"),
        log_path=os.path.join(log_dir, algorithm_name, env_id, "eval_logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        print(f"Training of {algorithm_name} on {env_id} finished.")
    except Exception as e:
        print(f"Error during training {algorithm_name} on {env_id}: {e}")
    finally:
        train_env.close()
        eval_env.close()

# --- Main Training Loop ---
if __name__ == "__main__":
    for env_id in ENVIRONMENTS:
        print(f"\n===== Starting experiments for environment: {env_id} =====")
        
        temp_env_config = {}
        if env_id == "racetrack-v0":
            temp_env_config = {"action": {"type": "DiscreteMetaAction"}}
        temp_env = create_env(env_id, **temp_env_config)
        env_action_type = get_action_space_type(temp_env)
        temp_env.close() # Close the temporary environment

        for algo_name, algo_config in ALGORITHMS.items():
            if (algo_config["action_space_type"] == "Discrete" and env_action_type == "Continuous") or \
               (algo_config["action_space_type"] == "Continuous" and env_action_type == "Discrete"):
                print(f"Skipping {algo_name} on {env_id} due to incompatible action spaces ({algo_config['action_space_type']} vs {env_action_type}).")
                continue

            # Simple hyperparameter grid search
            best_reward = -float('inf')
            best_hyperparams = None
            best_model_path = None

            hyperparam_combinations = [{}] # Start with empty dict for the first combination
            for param, values in algo_config["hyperparams"].items():
                new_combinations = []
                for current_combo in hyperparam_combinations:
                    for value in values:
                        new_combo = current_combo.copy()
                        new_combo[param] = value
                        new_combinations.append(new_combo)
                        # The following 'break' statements will cause it to only take
                        # the first value of the first hyperparameter from the list.
                        # Remove them if you intend a full grid search.
                        break # If you want only one value per param, keep this.
                    break # If you want only one combination, keep this.
                hyperparam_combinations = new_combinations
                break # If you want only one combination, keep this.

            print(f"\n--- Testing {len(hyperparam_combinations)} hyperparameter combinations for {algo_name} on {env_id} ---")

            for i, hparams in enumerate(hyperparam_combinations):
                run_log_dir = os.path.join(LOG_DIR, f"{algo_name}_{env_id}_run_{i}")
                os.makedirs(run_log_dir, exist_ok=True) # Ensure directory exists for this run
                
                # Check if the environment requires specific configurations for continuous/discrete actions
                env_config = {}
                if env_id == "racetrack-v0" and algo_name == "DQN":
                    print("INFO: Configuring racetrack-v0 for discrete actions for DQN.")
                    env_config = {"action": {"type": "DiscreteMetaAction"}}
                elif env_id == "racetrack-v0" and algo_name in ["PPO", "SAC"]:
                    print("INFO: Configuring racetrack-v0 for continuous actions for PPO/SAC.")
                    env_config = {"action": {"type": "ContinuousAction"}}
                
                run_name = f"{algo_name}-{env_id}-run-{i}"
                wandb_run = wandb.init(
                    project="ECE457C-RL-Assignment", 
                    entity="chacloutier-university-of-waterloo",    
                    sync_tensorboard=True,            # Sync SB3's TensorBoard logs to WandB
                    monitor_gym=True,                 # Monitor Gymnasium environments (optional, can capture videos)
                    name=run_name,
                    config={                          # Log all hyperparameters and run details
                        "algo": algo_name,
                        "env_id": env_id,
                        "run_index": i,
                        "total_timesteps": TOTAL_TIMESTEPS,
                        "eval_freq": EVAL_FREQ,
                        "n_eval_episodes": N_EVAL_EPISODES,
                        **hparams,                    # Unpack the current hyperparams
                        **env_config                  # Include environment specific configs
                    }
                )

                # Create the environment with potential config changes
                current_train_env = create_env(env_id, **env_config)
                # You can set render_mode="human" here to visualize during evaluation
                # current_eval_env = create_env(env_id, **env_config) 
                # current_eval_env = create_env(env_id, render_mode="human", **env_config) 
                raw_eval_env = create_env(env_id, render_mode="rgb_array", **env_config) 

                # MODIFIED: Wrap environments with Monitor for logging to the specific run directory
                current_train_env = Monitor(current_train_env, os.path.join(run_log_dir, "train_monitor.csv"))
                current_eval_env = Monitor(raw_eval_env, os.path.join(run_log_dir, "eval_monitor.csv"))

                # current_eval_env = RecordEpisodeStatistics(current_eval_env)
                video_dir = os.path.join(run_log_dir, "videos")
                os.makedirs(video_dir, exist_ok=True)

                # Only record every 10th episode, let wandb upload the video
                current_eval_env = RecordVideo(
                    current_eval_env,
                    video_folder=video_dir,  # dummy path, will be ignored by wandb
                    episode_trigger=lambda ep: ep % 10 == 0
                )

                model = algo_config["model"]("MlpPolicy", current_train_env, verbose=0, tensorboard_log=run_log_dir, **hparams)

                eval_callback = EvalCallback(
                    current_eval_env,
                    best_model_save_path=os.path.join(run_log_dir, "best_model"),
                    log_path=os.path.join(run_log_dir, "eval_logs"),
                    eval_freq=EVAL_FREQ,
                    n_eval_episodes=N_EVAL_EPISODES,
                    deterministic=True,
                    render=False,
                )
                # eval_callback = EvalCallback(
                #     current_eval_env,
                #     best_model_save_path=os.path.join(run_log_dir, "best_model"),
                #     log_path=os.path.join(run_log_dir, "eval_logs"),
                #     eval_freq=EVAL_FREQ,
                #     n_eval_episodes=N_EVAL_EPISODES,
                #     deterministic=True,
                #     render=True,
                # )

                wandb_callback = WandbCallback(
                    model_save_path=None, # EvalCallback handles best model saving locally
                    verbose=0, # Set to 1 for more verbose WandB logging
                )

                try:
                    # MODIFIED: Pass both callbacks as a list
                    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback, wandb_callback])
                    print(f"Training of {algo_name} on {env_id} finished.")
                    
                    # The EvalCallback saves the best model inside the specified best_model_save_path
                    # So, the actual path to the .zip file will be best_model_save_path/best_model.zip
                    best_model_save_dir = os.path.join(run_log_dir, "best_model") # Directory from EvalCallback
                    best_model_current_run_path = os.path.join(best_model_save_dir, "best_model.zip") # Actual .zip file path

                    if os.path.exists(best_model_current_run_path):
                        loaded_model = algo_config["model"].load(best_model_current_run_path)
                        mean_reward, _ = evaluate_policy(loaded_model, current_eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
                        print(f"  Run {i+1} HParams: {hparams}, Mean Reward (Best Model): {mean_reward:.2f}")

                        wandb.log({
                            "final/mean_reward": mean_reward,
                            "final/best_model_path": best_model_current_run_path
                        })

                        if mean_reward > best_reward:
                            best_reward = mean_reward
                            best_hyperparams = hparams
                            best_model_path = best_model_current_run_path
                            print(f"  New best found for {algo_name} on {env_id}: Mean Reward = {best_reward:.2f}")
                    else:
                        print(f"  No best model saved for Run {i+1}. This might happen if evaluation did not complete or reward was always decreasing or if the path is incorrect.")
                        wandb.log({"final/status": "no_best_model_saved"})
            
                    # Path where videos are saved
                    video_dir = os.path.join(run_log_dir, "videos")
                    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

                    # Upload and delete each video
                    for video_path in video_files:
                        # Log to wandb
                        wandb.log({"evaluation/video": wandb.Video(video_path, format="mp4")})

                        # Delete the video file
                        try:
                            os.remove(video_path)
                            print(f"Deleted local video file: {video_path}")
                        except Exception as e:
                            print(f"Could not delete {video_path}: {e}")

                except Exception as e:
                    print(f"  Error during training run {i+1} for {algo_name} on {env_id} with HParams {hparams}: {e}")
                    wandb.log({"final/status": "failed", "error_message": str(e)})
                    traceback.print_exc()
                finally:
                    current_train_env.close()
                    raw_eval_env.close()
                    wandb_run.finish()

            # The best_hyperparams tracking here is for the local script's output,
            # WandB will provide a much better overview of all runs.
            if best_hyperparams:
                print(f"\nBest hyperparameters for {algo_name} on {env_id}: {best_hyperparams} with mean reward {best_reward:.2f}")
                final_best_model_dir = os.path.join(LOG_DIR, "final_best_models", algo_name)
                os.makedirs(final_best_model_dir, exist_ok=True)
                final_best_model_path = os.path.join(final_best_model_dir, f"{env_id}_best_model.zip")
                if best_model_path and os.path.exists(best_model_path):
                    shutil.copyfile(best_model_path, final_best_model_path)
                    print(f"Saved best model to: {final_best_model_path}")
                else:
                    print(f"Could not find best model to save for {algo_name} on {env_id}.")
            else:
                print(f"No successful training runs for {algo_name} on {env_id}.")

    print("\n===== All experiments finished. =====")
    print(f"TensorBoard logs are available in: {LOG_DIR}")
    print("Run `tensorboard --logdir ./rl_logs` in your terminal to view results.")