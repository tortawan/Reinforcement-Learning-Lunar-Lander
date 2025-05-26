# DQN Agent for LunarLander-v3 using PyTorch

![](/images/LunarLander.gif)


This project implements a Deep Q-Network (DQN) agent to solve the `LunarLander-v3` environment from the [Gymnasium](https://gymnasium.farama.org/) library. The agent is built using PyTorch and aims to learn an optimal policy for landing the lunar module safely.

## 🎯 Project Overview

The Deep Q-Network (DQN) algorithm is a cornerstone of modern reinforcement learning, designed to learn optimal policies by approximating the Q*(s,a) function (optimal action-value function). This project applies DQN to the classic `LunarLander-v3` control problem. The agent observes the lander's state and learns to choose actions (fire main engine, left/right thrusters, or do nothing) to maximize cumulative rewards, ultimately aiming for a soft and accurate landing.

## ✨ Key Features

* **Deep Q-Network (DQN):** A neural network (`QNetwork` class) built with PyTorch approximates the Q-values.
* **Experience Replay:** Utilizes a replay buffer (`collections.deque`) to store and sample past experiences (state, action, reward, next_state, done), breaking correlations and stabilizing learning.
* **Target Network:** Employs a separate target network that is periodically updated to provide stable targets for Q-value updates, mitigating oscillations during training.
* **Epsilon-Greedy Exploration:** Balances exploration (trying new actions to discover better strategies) with exploitation (choosing the best-known actions). The exploration rate (epsilon) decays over the training period.
* **Periodic Evaluation:** The agent's learning progress is assessed at regular intervals by evaluating its performance on a set number of episodes without exploration or learning updates.
* **Early Stopping:** Training automatically concludes if the agent consistently achieves the predefined `TARGET_SCORE_AVG` during these periodic evaluations, ensuring efficiency.
* **Model Persistence:** Trained model weights can be saved to a `.pth` file and reloaded, allowing for continuation of training or for direct deployment/evaluation of a trained agent.
* **Progress Visualization:** Training progress, including scores per episode and evaluation scores, is plotted using Matplotlib and saved as an image (e.g., `lunar_lander_training_progress.png`).
* **Results Logging:** Detailed training statistics (episode number, scores, epsilon, etc.) are logged to a CSV file using Pandas for thorough analysis.
* **Configurable Hyperparameters:** Essential parameters such as network architecture, learning rate, batch size, replay buffer capacity, and discount factor (gamma) are clearly defined (e.g., in an `exp_spec` dictionary or as constants) for easy experimentation.

## 🛠️ Technology Stack

* Python 3.8+
* PyTorch
* Gymnasium (with Box2D, i.e., `gymnasium[box2d]`)
* NumPy
* Pandas
* Matplotlib

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tortawan/Reinforcement-Learning-Lunar-Lander.git](https://github.com/tortawan/Reinforcement-Learning-Lunar-Lander.git)
    cd Reinforcement-Learning-Lunar-Lander
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    Make sure you have `Box2D` installed as part of Gymnasium.
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` is up-to-date and includes `gymnasium[box2d]`)*



## 📊 Results & Performance

*(This section is crucial! Fill it with your actual results.)*

* **Training Environment:** `LunarLander-v3`
* **Target Score for Solving:** The environment is typically considered solved if the agent achieves an average reward of `[e.g., +200]` over 100 consecutive episodes.
* **Performance Achieved:**
    * The agent was trained for `[Number, e.g., 1500]` episodes.
    * It achieved an average reward of `[Your Average Reward, e.g., +230]` over the last 100 evaluation episodes.
    * The environment was successfully solved after approximately `[Number, e.g., 1200]` training episodes.
* **Best Score in a Single Episode:** `[Your Best Score]`
* **Training Duration:** Approximately `[Time, e.g., 2 hours on CPU/GPU specs]`
* **Key Hyperparameters Used:**
    * Learning Rate (`LR`): `[e.g., 5e-4]`
    * Batch Size: `[e.g., 64]`
    * Replay Buffer Size: `[e.g., 1e5]`
    * Discount Factor ($\gamma$): `[e.g., 0.99]`
    * Target Network Update Frequency (`TAU` or `UPDATE_EVERY`): `[e.g., 1e-3 or 4 episodes]`
    * Epsilon Decay Rate: `[e.g., 0.995]`
* **Training Progress Plot:**
    *(You can embed the training plot image here if you like, or link to it)*
    `![Training Progress](path_to_your_training_plot.png)`

## 🔮 Future Improvements & Ideas

* Implement **Double DQN (DDQN)** to address potential Q-value overestimation issues.
* Explore **Dueling DQN** architecture to separate the estimation of state values and advantage values.
* Experiment with prioritized experience replay (PER).
* Conduct more extensive hyperparameter tuning using automated methods (e.g., Optuna, Ray Tune).
* Compare performance with other RL algorithms like PPO or A2C on this environment.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
*(Create a `LICENSE.md` file in your repository with the MIT License text or another license of your choice.)*

---
