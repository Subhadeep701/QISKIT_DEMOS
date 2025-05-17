import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import plot_bloch_multivector,plot_histogram
from qiskit.primitives import StatevectorSampler
# Create pure states
qc=QuantumCircuit(2)
qc.x(0)
qc.s(1)
qc.cx(1,0)
#qc.x(1)
qc.h(0)
qc.h(1)

qc.draw("mpl")
print(Operator.from_circuit(qc))
plt.show()
