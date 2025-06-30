from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import UnitaryGate
from qiskit.visualization import plot_histogram
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# Number of counting qubits
n_count = 6

# Generate a random 4x4 unitary (acts on 2 qubits)
U = random_unitary(4).data
eigvals, eigvecs = np.linalg.eig(U)  # True eigenvalues/vectors (for comparison)

# Create the circuit: n_count counting qubits + 2 target qubits
qc = QuantumCircuit(n_count + 2, n_count)

# Step 1: Apply Hadamards to counting register
qc.h(range(n_count))

# Step 2: Prepare target qubits in |++> (equal superposition)
qc.h(n_count)
qc.h(n_count + 1)

# Step 3: Apply controlled-U^{2^j} from least to most significant bit
for j in range(n_count):
    power = 2 ** j
    U_power = np.linalg.matrix_power(U, power)
    controlled_U = UnitaryGate(U_power).control(1)  # Controlled version of U^power
    qc.append(controlled_U, [j] + [n_count, n_count + 1])

# Step 4: Apply inverse QFT to the counting register
def inverse_qft(qc, n):
    for j in range(n // 2):
        qc.swap(j, n - j - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-pi / (2 ** (j - m)), m, j)
        qc.h(j)

inverse_qft(qc, n_count)

# Step 5: Measure the counting register
qc.measure(range(n_count), range(n_count))

# Simulate the circuit
backend = Aer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc, backend)
job = backend.run(transpiled_qc, shots=500)
counts = job.result().get_counts()

# Sort and print top estimated phases
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
print("\n=== Top Estimated Phases from QPE ===")
for bitstring, freq in sorted_counts[:5]:
    decimal = int(bitstring, 2)
    phase = decimal / (2 ** n_count)
    eigenvalue = round(np.exp(2j * pi * phase), 5)
    print(f"Bitstring: {bitstring} | Phase: {phase:.5f} | Eigenvalue ≈ {eigenvalue}")

# Print true eigenvalues
print("\n=== True Eigenvalues of the Random 4x4 Unitary ===")
for i, val in enumerate(eigvals):
    phase = (np.angle(val) / (2 * pi)) % 1
    print(f"Eigenvalue {i}: {np.round(val, 5)} | Phase: {phase:.5f}")

# Plot the results
plot_histogram(counts)
plt.title("QPE Histogram (No Known Eigenvectors)")
plt.show()
