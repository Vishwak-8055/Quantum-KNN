from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def run_quantum_knn():

    qc = QuantumCircuit(2)

    qc.h(0)
    qc.cx(0,1)
    qc.measure_all()

    simulator = AerSimulator()

    result = simulator.run(qc).result()

    counts = result.get_counts()

    return counts