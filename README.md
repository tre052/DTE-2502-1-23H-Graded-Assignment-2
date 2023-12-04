# Running Graded assignment 02. 

### 1. Train the model (first runs 1 full iteration of BFS, then runs 300 iterations of DeepQ)

command: "python [training.py](../training.py)"

The trained model and its checkpoints are found in the /models/ folder

### 2. Test the model

command: "python [game_visualization.py](../game_visualization.py)"

Game visualizations are output into the /output/ folder

# Additional dependencies

None

# Snake Reinforcement Learning

Code for training a Deep Reinforcement Learning agent to play the game of Snake.
The agent takes 2 frames of the game as input (image) and predicts the action values for
the next action to take.
