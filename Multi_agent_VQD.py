import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorEstimator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import gym


# === Define the Quantum Environment ===
class QuantumEnv(gym.Env):
    def __init__(self):
        super(QuantumEnv, self).__init__()
        self.num_qubits = 2
        self.num_params = 2  # Number of tunable parameters
        self.params = ParameterVector("θ", self.num_params)
        self.target_energy = -1.0  # Target energy
        self.hamiltonian = Operator.from_label("ZZ") + Operator.from_label("XI")
        self.simulator = AerSimulator()
        self.estimator = StatevectorEstimator()

        # Define observation and action spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.num_params,), dtype=np.float32)
        self.action_space = spaces.Box(-0.1, 0.1, shape=(self.num_params,), dtype=np.float32)

        # Initialize the state
        self.reset()

    def reset(self):
        # Random initialization of parameters
        self.current_params = np.random.uniform(0, 2 * np.pi, self.num_params)
        return self.current_params.astype(np.float32)

    def step(self, action):
        # Update parameters with action
        self.current_params += action

        # Build and execute quantum circuit
        qc = QuantumCircuit(self.num_qubits)
        qc.rx(self.current_params[0], 0)
        qc.ry(self.current_params[1], 1)
        energy = self.estimator.run(qc, self.hamiltonian).result().values[0]

        # Compute reward (minimize energy)
        reward = -abs(self.target_energy - energy)

        # Check termination condition (optional for energy threshold)
        done = False

        # Return observation, reward, done, and info
        return self.current_params.astype(np.float32), reward, done, {"energy": energy}

    def render(self, mode="human"):
        print(f"Current Parameters: {self.current_params}")
        print(f"Energy: {self.estimator.run(self.current_params, self.hamiltonian).result().values[0]}")


# === Train the RL Agent ===
if __name__ == "__main__":
    env = DummyVecEnv([lambda: QuantumEnv()])  # Wrap environment for Stable-Baselines3

    # Initialize PPO Agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)  # Train the model

    # Test the trained agent
    env = QuantumEnv()  # Create a fresh environment
    obs = env.reset()

    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
