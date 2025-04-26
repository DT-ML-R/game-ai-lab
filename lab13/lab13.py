from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from types import MethodType
import os
import numpy as np
import matplotlib.pyplot as plt

# TODO Write your reward functions here
# Reward function 1: Distance-based reward
def myreward1(state):
    # Reward based on Manhattan distance to goal
    goal_state = 15
    current_row = state // 4
    current_col = state % 4
    goal_row = goal_state // 4
    goal_col = goal_state % 4
    
    # Manhattan distance to goal
    distance = abs(current_row - goal_row) + abs(current_col - goal_col)
    
    # Reward inversely proportional to distance
    if state == goal_state:
        return 10.0  # Large reward for reaching goal
    else:
        return 1.0 / (distance + 1)  # Smaller reward for being closer

# Reward function 2: Encourages exploration with center-based rewards
def myreward2(state):
    # Reward states closer to center higher to encourage exploration
    row = state // 4
    col = state % 4
    
    # Distance from center of grid (1.5, 1.5)
    center_distance = abs(row - 1.5) + abs(col - 1.5)
    
    if state == 15:  # Goal state
        return 10.0  # Large reward for reaching goal
    else:
        return 2.0 / (center_distance + 1)  # Higher reward for center states

# Reward function 3: Optimal path reward
def myreward3(state):
    # Rewards being on an optimal path to the goal
    # Define the optimal path
    optimal_path = [0, 1, 2, 3, 7, 11, 15]
    
    if state == 15:  # Goal state
        return 10.0  # Large reward for reaching goal
    elif state in optimal_path:
        return 1.0  # Smaller reward for being on optimal path
    else:
        return -0.1  # Small penalty for being off the path

# Default reward function to use (change to myreward1, myreward2, or myreward3)
def myreward(state):
    # Uncomment the one you want to use
    return myreward1(state)
    # return myreward2(state)
    # return myreward3(state)

# Create environment

print("Creating environment...")
env = gym.make('FrozenLake-v1', render_mode='rgb_array', desc=None, map_name="4x4", is_slippery=False)

# Wrapper to modify the reward 
def transition_function(self, action):
    # perform and update
    state, reward, done, truncated, info = self.internal_step(action)
    reward = myreward(state)  # Using the selected reward function
    return state, reward, done, truncated, info

# Transition function in gymnasium environments is called step
env.internal_step = env.step
env.step = MethodType(transition_function, env)

# Create directory for saving results
os.makedirs("./lab13/results", exist_ok=True)

# Instantiate the agent
print("Creating agent...")
# Completely disable tensorboard logging to avoid dependency issues
model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=None)

# Simple callback to track rewards
class RewardTracker:
    def __init__(self):
        self.rewards = []
        self.episode_lengths = []
        self.current_rewards = []
        self.current_length = 0
        
    def __call__(self, locals, globals):
        # Track rewards for each step
        reward = locals['rewards'][0]
        done = locals['dones'][0]
        
        self.current_rewards.append(reward)
        self.current_length += 1
        
        if done:
            # End of episode
            self.rewards.append(sum(self.current_rewards))
            self.episode_lengths.append(self.current_length)
            self.current_rewards = []
            self.current_length = 0
            
            # Print progress every 10 episodes
            if len(self.rewards) % 10 == 0:
                print(f"Episodes: {len(self.rewards)}, Avg Reward: {np.mean(self.rewards[-10:]):.2f}")
                
        return True
    
    def save_results(self, reward_function_name):
        # Save results to file
        filename = f"./lab13/results/{reward_function_name}_results.txt"
        with open(filename, 'w') as f:
            f.write("Episode,Reward,Length\n")
            for i, (r, l) in enumerate(zip(self.rewards, self.episode_lengths)):
                f.write(f"{i+1},{r},{l}\n")
        
        # Plot learning curve
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards)
        plt.title(f"Rewards - {reward_function_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title(f"Episode Lengths - {reward_function_name}")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        
        plt.tight_layout()
        plt.savefig(f"./lab13/results/{reward_function_name}_plot.png")
        plt.close()

# Create the reward tracker
reward_tracker = RewardTracker()

# Train the agent
print("Training agent... (this might take a while)")
model.learn(total_timesteps=int(1e4), progress_bar=False, callback=reward_tracker)

# Get the name of the current reward function
current_reward_func = "myreward1"  # Change to match
print(f"Using reward function: {current_reward_func}")

# Save the results
reward_tracker.save_results(current_reward_func)

# Save the agent
model.save("./lab13/ppo_frozenlake")
del model  # delete trained model to demonstrate loading

# Load the trained agent
print("Loading model...")
model = PPO.load("./lab13/ppo_frozenlake", env=env)

# Evaluate the agent
print("Evaluating agent...")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Enjoy trained agent
print("Running agent in environment...")
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")