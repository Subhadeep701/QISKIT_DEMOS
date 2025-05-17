from qiskit import QuantumCircuit,transpile
from qiskit.circuit.library.standard_gates.equivalence_library import qargs
from qiskit.quantum_info import Statevector,random_unitary
from qiskit.circuit.library import StatePreparation,UnitaryGate
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from matplotlib import pyplot as plt
import numpy as np
qc=QuantumCircuit(2,1)
qc.h(0)
b=StatePreparation([1/np.sqrt(2),1/np.sqrt(2)])
qc.append(b,[1])
unitary=random_unitary(2)
unitary=UnitaryGate(unitary).control(1)
qc.append(unitary,[0,1])
qc.h(0)
#qc.measure([0],[0])
counts=Statevector(qc).sample_counts(qargs=[0],shots=1000)
#print(f"a:{a}")
total_shots = sum(counts.values())
probabilities = {key: value / total_shots for key, value in counts.items()}
count_list=list(counts.values())
# Print results
print(f"Counts: {counts}")

print(f"Probabilities: {probabilities}")
print(f"psi*={np.conjugate(np.array(b.params))}, unitary={unitary.params[0]},psi={np.array(b.params)}, psi|u|psi={np.dot(np.dot(np.conjugate(b.params),unitary.params[0]),b.params)}")

qc.measure_all()
qc_t=transpile(qc,basis_gates=['u', 'cx'])
qc_t.draw("mpl")
Sampler=StatevectorSampler()
job=Sampler.run([qc_t],shots=1000).result()[0]
count_sampler=job.data.meas.get_counts()
g=list(count_sampler.values())

sum_values_0=sum(value for key,value in count_sampler.items() if key.endswith("0"))
probability=sum_values_0/sum(count_sampler.values())
print(f"Sampler_probability{probability}")

print(f"Statevectorsampler_estimated_hadamard_test={2*probability-1}")
print(f"True psi|u|psi={np.dot(np.dot(np.conjugate(b.params),unitary.params[0]),b.params)}")
print(f"sample_counts_hadamard_test={(count_list[0]-count_list[1])/total_shots}")
plt.show()