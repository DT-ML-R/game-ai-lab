# Lab 13 Reinforcement Learning Lab Report

## Custom Reward Function 1: Distance-Based Reward

The first reward function (myreward1) was designed to guide the agent toward the goal based on Manhattan distance. The formula was:

```python
if state == goal_state:
    return 10.0  # Large reward for reaching goal
else:
    return 1.0 / (distance + 1)  # Smaller reward for being closer
```

This approach creates a gradient of rewards across the grid, with states closer to the goal receiving higher rewards. The key characteristics of this reward function:

- Provides continuous feedback even when far from the goal
- Creates a clear gradient toward the goal state
- Gives a large bonus (10.0) for reaching the goal

### Learning Performance:
The distance-based reward function performed excellently, achieving a perfect evaluation score of 100.0 ± 0.0. The agent was able to consistently find the optimal path to the goal. The learning curve showed rapid improvement, with the agent reaching high performance within about 500-600 episodes.

## Custom Reward Function 2: Center-Based Exploration Reward

The second reward function (myreward2) was designed to encourage exploration by rewarding states closer to the center of the grid:

```python
if state == 15:  # Goal state
    return 10.0  # Large reward for reaching goal
else:
    return 2.0 / (center_distance + 1)  # Higher reward for center states
```

This approach has the following characteristics:
- Encourages the agent to explore the central area of the grid
- Provides higher rewards for states that might not be directly on the path to the goal
- Still maintains a high reward for reaching the goal state

### Learning Performance:
The center-based reward performed well, achieving an evaluation score of 50.0 ± 0.0. The learning curve showed a more gradual improvement compared to the distance-based reward:
- Started with average rewards around 5-10 for the first 200 episodes
- Gradually increased to 10-15 by episode 400
- Reached 25-30 by episode 560
- Finally achieved consistent rewards of 40+ by episode 590-610

This slower learning is expected, as the reward function initially encourages exploration of the center rather than directly heading to the goal.

## Custom Reward Function 3: Optimal Path Reward

The third reward function (myreward3) was designed to explicitly encourage following a predefined optimal path:

```python
if state == 15:  # Goal state
    return 10.0  # Large reward for reaching goal
elif state in optimal_path:
    return 1.0  # Smaller reward for being on optimal path
else:
    return -0.1  # Small penalty for being off the path
```

This approach:
- Directly rewards states on the optimal path
- Penalizes states off the optimal path
- Provides a large reward for reaching the goal

### Learning Performance:
Based on earlier testing, the optimal path reward function achieved a mean reward of 14.29 ± 0.00, which was surprisingly lower than both the distance-based and center-based rewards. The learning curve showed:
- Slow initial progress with rewards around 1-2
- Gradual improvement but never reaching the performance of the other reward functions
- Difficulty in learning the exact optimal path due to the sparse nature of the rewards

## Discussion and Comparison

The three reward functions demonstrated important principles in reinforcement learning:

1. **Distance-based reward (myreward1)** performed best because it provided a smooth gradient toward the goal. This created a clear learning signal that helped the agent understand which actions moved it closer to the goal, regardless of its current position.

2. **Center-based reward (myreward2)** encouraged exploration but took longer to learn the optimal path. While it eventually achieved good performance (50.0), it didn't match the perfect score of the distance-based reward. This shows that encouraging exploration can be beneficial, but might slow down convergence to the optimal solution.

3. **Optimal path reward (myreward3)** had the most challenging learning task despite seeming the most direct. The sparse reward signal (only rewards on specific states) made it difficult for the agent to learn the correct behavior when off the optimal path. This demonstrates the "sparse reward problem" in reinforcement learning.

The results highlight the importance of reward shaping in reinforcement learning. A well-designed reward function that provides continuous feedback and creates a gradient toward the desired behavior (like myreward1) can lead to faster and more effective learning than one that only rewards specific states or behaviors (like myreward3).

In real-world applications, designing effective reward functions is often more important than choosing sophisticated algorithms. This lab demonstrates how different reward functions can dramatically affect an agent's learning performance, even when using the same learning algorithm (PPO) and environment.