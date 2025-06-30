from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit.library import MCMT,ZGate
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator,SparsePauliOp
from qiskit.primitives import StatevectorEstimator,StatevectorSampler
from matplotlib import pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import math

import math
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


def grover_oracle(marked_states):
    """
    Build a Grover oracle for multiple marked states.

    Parameters:
        marked_states (list[str]): Marked states of the oracle.

    Returns:
        QuantumCircuit: Quantum circuit representing Grover oracle.
    """
    if not isinstance(marked_states, list):
        marked_states = [marked_states]

    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits)

    for target in marked_states:
        rev_target = target[::-1]

        # Handle the all-1 state separately
        if all(bit == "1" for bit in rev_target):
            # Directly apply multi-controlled Z gate
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        else:
            # Find the indices of all '0' elements in the bit-string
            zero_inds = [i for i, bit in enumerate(rev_target) if bit == "0"]

            # Apply X gates to flip '0' bits
            qc.x(zero_inds)

            # Apply multi-controlled Z gate
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)

            # Undo the X gates
            qc.x(zero_inds)

        qc.barrier()

    return qc

def apply_reflection_about_mean(qc, n):
    """
    Apply the reflection about the mean (Grover diffuser).

    Parameters:
        qc : QuantumCircuit
            The quantum circuit to which the reflection will be applied.
        n : int
            Number of qubits.
    """
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)  # Multi-controlled NOT targeting the last qubit
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


# Parameters
n = 3
marked_states = ["110","111"]

# Build the Grover oracle
oracle = grover_oracle(marked_states)

# Create the main circuit
qc = QuantumCircuit(n, n)
qc.h(range(n))  # Apply Hadamard gates to create uniform superposition

# Apply the Grover oracle
qc.compose(oracle, inplace=True)
qc.barrier()

# Apply the reflection about the mean
apply_reflection_about_mean(qc, n)

# Calculate the optimal number of iterations
optimal_num_iterations = math.floor(
    math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2**n)))
)

# Repeat the Grover operator
for _ in range(optimal_num_iterations - 1):
    qc.compose(oracle, inplace=True)
    qc.barrier()
    apply_reflection_about_mean(qc, n)

# Add measurement
qc.measure(range(n), range(n))

# Visualize the circuit
qc.draw("mpl")

# Simulate the circuit
simulator = AerSimulator()
result = simulator.run(qc, shots=1000).result()
counts = result.get_counts()

# Plot the results
plot_histogram(counts)
plt.show()
