from qiskit import QuantumCircuit
from matplotlib import pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal,NLocal,RXGate
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp,Operator
from scipy.optimize import minimize
H =-np.array([[ 0.12, 0.00, 0.00, 0.00],
 [0, 0.45, 0.92, 0.00],
 [ 0.00, 0.92, -0.77, 0.34],
 [ 0.00,0.00, 0.34, 0.22]])
[EIG,EV]=np.linalg.eig(H)
print(EIG)
op=Operator(H)
A=SparsePauliOp.from_operator(op)
#print(A)
#print(np.log2(op.dim[1]))
#b=SparsePauliOp.to_matrix(A)
#print(b)
qubits=int(np.log2(op.dim[1]))

#print(*A.paulis,A)
a=TwoLocal(qubits,["rz","ry"],"cx",entanglement="linear",reps=1)
theta=np.random.rand(a.num_parameters)
#theta = (2 * np.pi * np.random.rand(8)).tolist()
print(theta)
qc=QuantumCircuit(qubits)
qc.x(0)
qc=qc.compose(a)
#qc.decompose().draw('mpl')



def cost_fun(params,ansatz,observable,estimator):
    pub=(ansatz,observable,[params])
    cost=estimator.run([pub]).result()[0].data.evs[0]
    return cost


estimator=StatevectorEstimator()
x0 = theta
result= minimize(cost_fun, x0, args=( qc,A, estimator), method="COBYLA",options=dict(maxiter=50))
print(result)


plt.show()

