from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
import numpy as np
from matplotlib import pyplot as plt
qc=QuantumCircuit(3)
qc.h(0)
print(f"{(lambda x: x / np.linalg.norm(x))(np.random.randn(2))}")
v=(lambda x:x/np.linalg.norm(x))(np.random.randn(2))
w=(lambda x:x/np.linalg.norm(x))(np.random.randn(2))
###in the second qubit put the statevector w
# to see the effect that 0 state probability is not 1
qc.append(StatePreparation(v),qargs=[1])
qc.append(StatePreparation(v),qargs=[2])
#qc.append(StatePreparation(w),qargs=[2])
qc.cswap(0,1,2)
qc.h(0)
qc.draw("mpl")
b=Statevector(qc).sample_counts(qargs=[0],shots=1000)
print(b)
count_0=sum(value for key,value in b.items() if key.startswith("0"))
print(f"Probability of 0: {count_0/sum(b.values())}")
plt.show()

