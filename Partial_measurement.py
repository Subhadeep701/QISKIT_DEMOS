from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector,Operator,SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from matplotlib import pyplot as plt
import numpy as np
###In this code, we do partial measurement of a two qubit system, we measure the 2nd bit,
# and calculate the probability of the 2nd bit to be 0 . The measurement operator
#is I @pi_0 that is performed through estimator primitive
b=np.array([1/np.sqrt(3),np.sqrt(2)/np.sqrt(3)])
c=np.outer(np.conjugate(b),b)
print(c)
pi_0=c
HIGH_OPERATOR=SparsePauliOp.from_operator(Operator(np.array([[1 ,0],[0,1]])).tensor(pi_0))
print(type(HIGH_OPERATOR))
print(HIGH_OPERATOR)
qc=QuantumCircuit(2)
qc.append(StatePreparation([np.sqrt(2)/3,-1/3,np.sqrt(2)/3,2/3]),[0,1])
estimator=StatevectorEstimator()
job=estimator.run([(qc,HIGH_OPERATOR)]).result()
print(job[0].data.evs)
qc.draw("mpl")
plt.show()
