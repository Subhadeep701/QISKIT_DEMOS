import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
# === Problem Setup ===
k = 2  # Number of eigenstates/agents
num_qubits = 2  # Qubits per circuit
alpha = 1.0  # Energy weight in reward
base_rewards = np.linspace(1.0, 0.5, k)  # Ranks: higher rank -> higher reward

# Define Hermitian matrix (Hamiltonian)
def generate_hermitian_matrix(n):
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    return (A + A.conj().T) / 2

n = 4
hermitian_matrix = generate_hermitian_matrix(n)
H = SparsePauliOp.from_operator(Operator(hermitian_matrix))

# Compute real eigenvalues of the Hamiltonian
real_eigenvalues = np.linalg.eigh(hermitian_matrix)[0]

# === Variational Ansatz for Each Agent ===
def make_agent_circuit(agent_id):
    ansatz = TwoLocal(num_qubits, ["rx", "ry"], "cx", entanglement="linear", reps=1)
    params = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
    return ansatz, params

agents = [make_agent_circuit(i) for i in range(k)]

# === Multi-Agent RL Environment ===
# class MultiAgentVQEEnv:
#     def __init__(self, agents, H):
#         self.agents = agents
#         self.H = H
#         self.estimator = StatevectorEstimator()
#         self.reset()
#
#     def reset(self):
#         self.thetas = {f"agent_{i}": np.random.uniform(0, 2 * np.pi, self.agents[i][0].num_parameters)
#                        for i in range(len(self.agents))}
#         return self.thetas
#
#     def step(self, actions):
#         # Update parameters based on actions
#         for agent_id, delta in actions.items():
#             self.thetas[agent_id] += delta
#
#         # Calculate energies
#         energies = {}
#         for i, (ansatz, _) in enumerate(self.agents):
#             agent_id = f"agent_{i}"
#             theta = self.thetas[agent_id]
#             bound_circuit = ansatz.assign_parameters(theta)
#             job = self.estimator.run([(bound_circuit, self.H)]).result()[0]
#             energies[agent_id] = np.real(job.data.evs)
#
#         # Rank energies and assign rewards
#         sorted_agents = sorted(energies, key=energies.get)
#         ranks = {agent: rank for rank, agent in enumerate(sorted_agents)}
#         rewards = {agent: base_rewards[ranks[agent]] - alpha * energies[agent]
#                    for agent in self.thetas}
#
#         return self.thetas, rewards, energies
#
# # === Training ===
# env = MultiAgentVQEEnv(agents, H)
#
# # Initialize RL policy (simple gradient update)
# def policy_update(theta, reward, delta, lr=0.1):
#     return theta + lr * reward * delta
#
# # Training loop
# num_epochs = 100
# rl_estimated_eigenvalues = []
# for epoch in range(num_epochs):
#     actions = {f"agent_{i}": np.random.uniform(-0.1, 0.1, agents[i][0].num_parameters)
#                for i in range(k)}
#     thetas, rewards, energies = env.step(actions)
#
#     # Policy update
#     for agent_id, reward in rewards.items():
#         delta = actions[agent_id]
#         env.thetas[agent_id] = policy_update(env.thetas[agent_id], reward, delta)
#
#     # Collect RL-estimated eigenvalues at the final epoch
#     if epoch == num_epochs - 1:
#         rl_estimated_eigenvalues = [energies[f"agent_{i}"] for i in range(k)]
#
#     # Logging
#     print(f"Epoch {epoch}: Energies = {energies}, Rewards = {rewards}")

# === Compare RL-Estimated and Real Eigenvalues ===
# real_eigenvalues_sorted = sorted(real_eigenvalues)
class CooperativeMultiAgentVQEEnv:
    def __init__(self, agents, H, lambda_=10.0):
        self.agents = agents
        self.H = H
        self.lambda_ = lambda_  # Weight for orthogonality penalty
        self.estimator = StatevectorEstimator()
        self.reset()

    def reset(self):
        self.thetas = {
            f"agent_{i}": np.random.uniform(0, 2 * np.pi, self.agents[i][0].num_parameters)
            for i in range(len(self.agents))
        }
        return self.thetas

    def evaluate_energy(self, agent_idx, theta):
        ansatz, _ = self.agents[agent_idx]
        bound = ansatz.assign_parameters(theta)
        job = self.estimator.run([(bound, self.H)]).result()[0]
        return float(np.real(job.data.evs))

    def calculate_total_energy(self):
        total_energy = 0
        for i in range(len(self.agents)):
            theta = self.thetas[f"agent_{i}"]
            total_energy += self.evaluate_energy(i, theta)
        return total_energy

    def calculate_orthogonality_penalty(self):
        penalty = 0
        keys = list(self.thetas.keys())
        for i, key_i in enumerate(keys):
            for j, key_j in enumerate(keys):
                if i < j:
                    # Compute inverse parameter distance
                    theta_i = self.thetas[key_i]
                    theta_j = self.thetas[key_j]
                    distance = np.linalg.norm(theta_i - theta_j)
                    penalty += 1 / (distance + 1e-6)  # Avoid division by zero
        return penalty

    def calculate_cooperative_loss(self):
        total_energy = self.calculate_total_energy()
        orthogonality_penalty = self.lambda_ * self.calculate_orthogonality_penalty()
        return total_energy + orthogonality_penalty


# === Training with SciPy ===
num_epochs = 50
env = CooperativeMultiAgentVQEEnv(agents, H, lambda_=1.0)

for epoch in range(num_epochs):
    # Optimize all agents cooperatively
    for i, (ansatz, _) in enumerate(agents):
        agent_id = f"agent_{i}"
        theta0 = env.thetas[agent_id]

        # Define the cooperative objective function for this agent
        def cooperative_obj_fn(theta):
            # Temporarily update the theta of this agent
            original_theta = env.thetas[agent_id]
            env.thetas[agent_id] = theta

            # Calculate the loss (total energy + orthogonality penalty)
            loss = env.calculate_cooperative_loss()

            # Restore the original theta
            env.thetas[agent_id] = original_theta
            return loss

        # Optimize with SciPy
        result = minimize(
            fun=cooperative_obj_fn,
            x0=theta0,
            method="L-BFGS-B",
            options={"maxiter": 1}
        )
        env.thetas[agent_id] = result.x

    # Log cooperative loss and energies
    cooperative_loss = env.calculate_cooperative_loss()
    print(f"Epoch {epoch:03d}: Cooperative Loss = {cooperative_loss}")

# Collect final estimated eigenvalues
final_energies = {f"agent_{i}": env.evaluate_energy(i, env.thetas[f"agent_{i}"]) for i in range(len(agents))}
print("\n=== Eigenvalues Comparison ===")
print(f"{'Agent':<10}{'RL Estimated':<20}{'Real Eigenvalue':<20}")
for i in range(len(agents)):
    print(f"{i:<10}{final_energies[f'agent_{i}']:<20}{real_eigenvalues[i]:<20}")
