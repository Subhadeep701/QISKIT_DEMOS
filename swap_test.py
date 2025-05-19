from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Operator,Statevector
import numpy as np
qc=QuantumCircuit(3)
qc.h(0)
qc.x(1)
qc.h(1)
qc.x(2)
qc.h(2)
qc.cswap(0,1,2)
qc.h(0)
b=Statevector(qc).sample_counts(qargs=[0],shots=1000)
print(b)
count_0=sum(value for key,value in b.items() if key.startswith("0"))
print(f"Probability of 0: {count_0/sum(b.values())}")


