import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp

# --- Parameters ---
n_qubits = 4
state_index = 1  # Can change to test other eigenstates

# --- Helper: Compute Z Correlation Matrix ---
def correlation_matrix_z(statevector, n):
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Zi = SparsePauliOp.from_list([("I" * i + "Z" + "I" * (n - i - 1), 1)])
            Zj = SparsePauliOp.from_list([("I" * j + "Z" + "I" * (n - j - 1), 1)])
            Zij = Zi @ Zj

            exp_Zi = np.real(statevector.expectation_value(Zi))
            exp_Zj = np.real(statevector.expectation_value(Zj))
            exp_ZiZj = np.real(statevector.expectation_value(Zij))
            C[i, j] = exp_ZiZj - exp_Zi * exp_Zj
    return C

# --- Case 1: Banded Z-only Hamiltonian (localized, uncorrelated) ---
banded_terms = [
    ("ZZII", 1.0),
    ("IZZI", 1.0),
    ("IIZZ", 1.0)
]
H_banded = SparsePauliOp.from_list(banded_terms)
eigvals_b, eigvecs_b = np.linalg.eigh(H_banded.to_matrix())
state_b = Statevector(eigvecs_b[:, state_index].copy())
C_banded = correlation_matrix_z(state_b, n_qubits)

# --- Case 2: Extended with X terms (delocalized, entangled) ---
entangled_terms = banded_terms + [
    ("XIII", 0.8),
    ("IXII", 0.8),
    ("IIXI", 0.8),
    ("IIIX", 0.8)
]
H_entangled = SparsePauliOp.from_list(entangled_terms)
eigvals_e, eigvecs_e = np.linalg.eigh(H_entangled.to_matrix())
state_e = Statevector(eigvecs_e[:, state_index].copy())
C_entangled = correlation_matrix_z(state_e, n_qubits)

# --- Print matrices ---
print("\n[Pauli-Z Correlation Matrix] Banded Hamiltonian:")
print(np.round(C_banded, 3))
print("\n[Pauli-Z Correlation Matrix] Entangled Hamiltonian:")
print(np.round(C_entangled, 3))

# --- Plot both ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im1 = axs[0].imshow(C_banded, cmap='coolwarm', vmin=-1, vmax=1)
axs[0].set_title("Banded (Z-only) Hamiltonian")
axs[0].set_xlabel("Z_j")
axs[0].set_ylabel("Z_i")
axs[0].set_xticks(range(n_qubits))
axs[0].set_yticks(range(n_qubits))

im2 = axs[1].imshow(C_entangled, cmap='coolwarm', vmin=-1, vmax=1)
axs[1].set_title("With X (Entangled) Hamiltonian")
axs[1].set_xlabel("Z_j")
axs[1].set_ylabel("Z_i")
axs[1].set_xticks(range(n_qubits))
axs[1].set_yticks(range(n_qubits))

plt.colorbar(im1, ax=axs[:], shrink=0.8, label="Pauli-Z Correlation")
plt.tight_layout()
plt.show()
