import qiskit.quantum_info as qi
import numpy as np

import qiskit_ibm_runtime
import qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, transpile
from qiskit.circuit.library import GroverOperator, QFT, GlobalPhaseGate, PhaseOracle, Diagonal
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Batch
from qiskit.algorithms import AmplitudeEstimation, EstimationProblem

class DistributedGroverAlgorithm:
    """
    DistributedGroverAlgorithm - class that combines everything needed for the Distributed Grover's Algorithm.
    WARNING! If you want to rerun the algorithm with a new oracle - create a new object of DistributedGroverAlgorithm
    and initialize it with a new oracle. Replacing oracle field will lead to problems!
    """
    def __init__(self, oracle=None):
        self.oracle = oracle
        self.oracle_matrix = None

    def get_oracle_matrix(self):
        if self.oracle_matrix == None:
            self.oracle_matrix = np.array(qi.Operator(self.oracle).data)                                                                                                  
        return self.oracle_matrix

    def _extract_subfunctions_from_matrix(self, oracle_matrix, k, bit_code=""):
        # Extract subfunctions
        to_extract0 = list(range(0, len(oracle_matrix), 2))
        to_extract1 = list(range(1, len(oracle_matrix), 2))
        oracle0_matrix = oracle_matrix[to_extract0][:, to_extract0]
        oracle1_matrix = oracle_matrix[to_extract1][:, to_extract1]

        # End condition reached
        if k == 1:
            return {"0" + bit_code : oracle0_matrix, "1" + bit_code : oracle1_matrix}
    
        # Continue on splitting
        return {**self._extract_subfunctions_from_matrix(oracle0_matrix, k - 1, bit_code="0" + bit_code), **self._extract_subfunctions_from_matrix(oracle1_matrix, k - 1, bit_code="1" + bit_code), }

    def extract_functions(self, k, bit_code=""):
        # Check if oracle is initialized, if not - no algorithm can be performed
        if self.oracle == None:
            raise ValueError("Oracle is None - No oracle specified") 

        # Check if oracle matrix is initialized, if not - create
        if self.oracle_matrix == None:
            self.get_oracle_matrix()

        #Extract subfunctions
        self.subfunctions = self._extract_subfunctions_from_matrix(self.oracle_matrix, k)
        return self.subfunctions

    @staticmethod
    def grover_operator_for_counting(n_iterations, oracle):
        grover_it = GroverOperator(oracle).repeat(n_iterations).to_gate()
        grover_it.label = f"Grover$^{n_iterations}$"
        return grover_it

    @staticmethod
    def calculate_t(y, c, n):
        # Calculate Theta
        theta = (y/(2**c))*np.pi*2
        print("Theta = %.5f" % theta)
        
        # Calculate No. of Solutions
        N = 2**n
        M = N * (np.sin(theta/2)**2)
        print(f"No. of Solutions = {M:.1f}")
        
        # Calculate Upper Error Bound
        m = c - 1 #Will be less than this (out of scope) 
        err = (np.sqrt(2*M*N) + N/(2**(m+1)))*(2**(-m))
        print("Error < %.2f" % err)
        return M

    @staticmethod
    def construct_quantum_counting_circuit(oracle, num_control_qubits, num_var_qubits):
        # Initialize registers and circuit
        control_qubits = QuantumRegister(num_control_qubits, name='c')
        var_qubits = QuantumRegister(num_var_qubits, name='v')
        cbits = ClassicalRegister(num_control_qubits, name='cbits')
        qc = QuantumCircuit(control_qubits, var_qubits, cbits)

        qc.h(var_qubits) #We use hadamard gates as our quantum algorithm A on the second register
        
        # Put everything into QFT basis
        qc.append(QFT(num_control_qubits), control_qubits)

        print("start repeating")
        # Add a bunch of Grover's operators
        n_iterations = 1
        grover_it = GroverOperator(oracle).repeat(1)
        for qubit in range(num_control_qubits):
            print(f"repeat {n_iterations}")
            qc.append(grover_it.to_gate().control(), [qubit] + list(range(num_control_qubits, num_var_qubits+num_control_qubits)))
            n_iterations *= 2
            grover_it = grover_it.repeat(2)
        print(f"finish repeating")
        # Revert QFT basis
        qc.append(QFT(num_control_qubits).inverse(), control_qubits)

        qc.measure(control_qubits, cbits)
        return qc

    @staticmethod
    def quantum_counting_qiskit(oracle, backend, num_var_qubits, num_control_qubits, tag):
        prep = QuantumCircuit(num_var_qubits)
        prep.h(list(range(num_var_qubits)))

        grover_op = GroverOperator(oracle)

        problem = EstimationProblem(
            state_preparation=prep,  # A operator
            grover_operator=grover_op,  # Q operator
            objective_qubits=list(range(num_var_qubits)) 
        )

        sampler = Sampler(backend=backend)

        ae = AmplitudeEstimation(
            num_eval_qubits=num_control_qubits,  # the number of evaluation qubits specifies circuit width and accuracy
            sampler=sampler,
        )

        ae_result = ae.estimate(problem)

        print(f"QISKIT RESULT MLE for {tag}: ", ae_result.mle * (2**num_var_qubits))
        print(f"QISKIT RESULT ESTIMATION for {tag}: ", ae_result.estimation * (2**num_var_qubits))
        return ae_result.mle * (2**num_var_qubits), ae_result.estimation * (2**num_var_qubits)

    @staticmethod
    def execute_on_the_backend(backend=None, list_qc=None, shots=None, tags=None):
        if backend == None:
            raise ValueError("No backend was provided")
        if list_qc == None or not isinstance(list_qc, list):
            raise ValueError("No quantum circuit to run was provided")
        if tags == None or not isinstance(tags, list):
            raise ValueError("Tags should be list")

        if isinstance(backend, qiskit_aer.backends.aer_simulator.AerSimulator):
            print("Using AerSimulator")
            list_transpiled_qc = [transpile(qc, backend) for qc in list_qc]
            job = backend.run(list_transpiled_qc, shots=shots, job_tags=tags)
            return job
            
        if isinstance(backend, qiskit_ibm_runtime.ibm_backend.IBMBackend):
            if shots == None:
                raise ValueError("Number of shot was not provided")
            if shots > 1024:
                print("WARNING! USING LARGE NUMBER OF SHOTS!")

            print("WARNING! USING IBM BACKEND!")
            list_transpiled_qc = [transpile(qc, backend) for qc in list_qc]
            job = backend.run(list_transpiled_qc, shots=shots, job_tags=tags)
            print(job.result())
            #dist = job.result().get_counts()
            return job

        raise ValueError("Backend was not recognized")

    @staticmethod        
    def counting_diagnostics(hist, num_control_qubits, num_var_qubits):
        #Get results
        measured_str = max(hist, key=hist.get)
        measured_int = int(measured_str, 2)
        print("Register Output = %i" % measured_int)
        t_value = DistributedGroverAlgorithm.calculate_t(measured_int, num_control_qubits, num_var_qubits)
        print(t_value)
        print("\n")
        # print("Circuit depth = %i" % QuantumCircuit.depth())

        return t_value, measured_int

    def set_oracle(self, oracle):
        self.oracle = oracle
        self.oracle_matrix = None

    def perform_algorithm(self, k=1, backend=None, shots=None):
        if backend == None:
            raise ValueError("backend is None - No backend was specified")
        if self.oracle == None:
            raise ValueError("Oracle is None - No oracle was specified") 
        
        # Step 1 - get all subfunctions
        self.extract_functions(k)

        
        
        #print(self.subfunctions)
        solved = []
        subfunction_list = []
        to_reverse = []

        
        keys_to_check = list([key for key in self.subfunctions.keys()])
        new_keys_to_check = []
        # Step 2 and 3
        # For each subfunciton construct an oracle and apply counting to it
        while len(keys_to_check) != 0:
            qc_list = []
            for key in keys_to_check:
                subfunction_matrix = self.subfunctions[key]
                num_qubits_subfunction = self.oracle.num_qubits - len(key)

                # If only one qubit left check if it's a solution manually
                if num_qubits_subfunction == 1:
                    real_part = np.array(subfunction_matrix.real)
                    if real_part[0][0] < 0:
                        solved.append("0" + key)
                    if real_part[1][1] < 0:
                        solved.append("1" + key)
                    continue
                subfunction_oracle = QuantumCircuit(num_qubits_subfunction)
                subfunction_oracle.unitary(subfunction_matrix, list(range(num_qubits_subfunction)))
                qc_list.append(subfunction_oracle)
            if len(qc_list) != 0:
                # And perform a counting operation
                for i in range(len(keys_to_check)):
                    
                    key = keys_to_check[i]
                    num_qubits_subfunction = self.oracle.num_qubits - len(key)
                    num_control_qubits = int(np.round(num_qubits_subfunction / 2))
                    print(num_control_qubits, num_qubits_subfunction)
                    subfunction_matrix = self.subfunctions[key]
                    subfunction_oracle = QuantumCircuit(num_qubits_subfunction)
                    subfunction_oracle.unitary(subfunction_matrix, list(range(num_qubits_subfunction)))
                    print(f"Counting for oracle with code {key}")
                    t, _ = DistributedGroverAlgorithm.quantum_counting_qiskit(subfunction_oracle, backend, num_qubits_subfunction, num_control_qubits, tag=f"qiskit_estimation_{num_qubits_subfunction}_{key}")

                    if np.round(t) == 2**(num_qubits_subfunction - 1):
                        # If number of solutions is equal to half - perform an extra split and them to queuem, 
                        # so later we iterate over them
                        print("Half of inputs, splitting more")
                        split_in_2 = self._extract_subfunctions_from_matrix(subfunction_matrix, 1, bit_code=key)
                        for key in split_in_2:
                            self.subfunctions[key] = split_in_2[key]
                            new_keys_to_check.append(key)
                    elif np.round(t) == 2**(num_qubits_subfunction):
                        # If all possible inputs are answers - add them to the answer list 
                        print("All possible inputs are answers, solving by default")
                        for i in range(int(np.round(t))):
                            solved.append(np.binary_repr(i, width=num_qubits_subfunction) + key)
                    elif np.round(t) > 2**(num_qubits_subfunction - 1):
                        print("Inverting an oracle")
                        # If more than half of inputs are answers - invert oracle, add flag to consider when calculating grover's
                        subfunction_oracle.append(GlobalPhaseGate(np.pi))
                        subfunction_list.append((subfunction_oracle, int(2**(num_qubits_subfunction) - np.round(t)), key))
                        to_reverse.append(key)
                    elif np.round(t) != 0:
                        print("Adding as is")
                        subfunction_list.append((subfunction_oracle, int(np.round(t)),key))
            while len(keys_to_check) != 0:
                keys_to_check.pop(0)
            while len(new_keys_to_check) != 0:
                keys_to_check.append(new_keys_to_check[0])
                new_keys_to_check.pop(0)
            
        return subfunction_list, solved, to_reverse

def run_counting(oracle, backend, num_var_qubits, num_control_qubits, tag):
    shots = 1024
    count_subfunction_qc = DistributedGroverAlgorithm.construct_quantum_counting_circuit(oracle, num_control_qubits, num_var_qubits)
    execution_result = DistributedGroverAlgorithm.execute_on_the_backend(backend=backend, list_qc=[count_subfunction_qc], shots=shots, tags=["estimation_experiment", tag])
    t, _ = DistributedGroverAlgorithm.counting_diagnostics(execution_result.result().get_counts(), num_control_qubits, num_var_qubits)


    # From qiskit

    prep = QuantumCircuit(num_var_qubits)
    prep.h(list(range(num_var_qubits)))

    grover_op = GroverOperator(oracle)

    problem = EstimationProblem(
        state_preparation=prep,  # A operator
        grover_operator=grover_op,  # Q operator
        objective_qubits=list(range(num_var_qubits)) 
    )

    sampler = Sampler(backend=backend)

    ae = AmplitudeEstimation(
        num_eval_qubits=num_control_qubits,  # the number of evaluation qubits specifies circuit width and accuracy
        sampler=sampler,
    )

    ae_result = ae.estimate(problem)
    print(f"QISKIT RESULT MLE for {tag}: ", ae_result.mle * (2**num_var_qubits))
    print(f"QISKIT RESULT ESTIMATION for {tag}: ", ae_result.estimation * (2**num_var_qubits))
    print("\n\n\n\n")
    return (int(np.round(t)), int(np.round(ae_result.mle * (2**num_var_qubits))))





def run_algo(oracle, backend, num_var_qubits, k, tag):
    def pre_process_results(subsolution, result, num_solutions, num_qubits_subfunction, results, solved, keylist):
        # Pick top num_solutions options and add the to dict of results
        if subsolution not in to_reverse:
            result_sorted = list(dict(sorted(result.items(), key=lambda item: item[1], reverse=True)[:int(np.round(num_solutions))]).keys())
            results.append(result_sorted)
            
        else:
            temp = []
            incorrect_solutions = list(dict(sorted(result.items(), key=lambda item: item[1], reverse=True)[:int(np.round(num_solutions))]).keys())
            for i in range(2**num_qubits_subfunction):
                bin_repr = str(np.binary_repr(i, width=num_qubits_subfunction))
                if bin_repr in incorrect_solutions:
                    continue
                temp.append(bin_repr)
            results.append(temp)

        keylist.append(subsolution)
        

    num_control_qubits = int(np.ceil((num_var_qubits - k)/2))

    num_qubits_subfunction = num_var_qubits - k
    
    dga = DistributedGroverAlgorithm(oracle=oracle)

    # Get subfunction oracles and number of solutions t
    #backend = Aer.get_backend('aer_simulator')
    subfunction_list, solved, to_reverse  = dga.perform_algorithm(k, backend=backend, shots=2048)

    to_reverse = set(to_reverse)

    #print(subfunction_list)
    results = []
    keylist = []
    list_qc = []
    for subfunction_oracle, num_solutions, subsolution in subfunction_list:
        print(num_var_qubits)
        print(subsolution)
        print(num_solutions)
    
        num_qubits_subfunction = num_var_qubits - len(subsolution)

        # Initialize circuit
        subfunction_qubits = QuantumRegister(num_qubits_subfunction, name='q')
        subfunction_cbits = ClassicalRegister(num_qubits_subfunction, name='cbits')
        qc = QuantumCircuit(subfunction_qubits, subfunction_cbits)
        qc.h(subfunction_qubits)
        # Estimate number of required iterations and aply grover's operator that many times
        n_iterations = np.pi/4 * np.sqrt((2**num_qubits_subfunction) / num_solutions)
        grover = DistributedGroverAlgorithm.grover_operator_for_counting(int(np.floor(n_iterations)), subfunction_oracle)
        # Append previous part to the circuit and measture the qubits
        qc.append(grover, subfunction_qubits)
        qc.measure(subfunction_qubits, subfunction_cbits)
        
        # Run constructed qc on the simulator
        transpiled_qc = transpile(qc, backend)
        list_qc.append(transpiled_qc)

    if len(subfunction_list) != 0:
        result = DistributedGroverAlgorithm.execute_on_the_backend(backend=backend, list_qc=list_qc, shots=2048,tags=["grover_part_"+subsolution, tag]).result().get_counts()

    if len(subfunction_list) == 1:
        result = [result]

    for i in range(len(subfunction_list)):
        num_qubits_subfunction = num_var_qubits - len(subfunction_list[i][2])
        print(subfunction_list[i][2], result[i])
        pre_process_results(subfunction_list[i][2], result[i], subfunction_list[i][1], num_qubits_subfunction, results, solved, keylist)

    print(results)    
    solutions = []
    for i in range(len(keylist)):
        for el in results[i]:
            solutions.append(str(el)+str(keylist[i]))
    print("Used grovers:", solutions)
    print("Found through splitting:", solved)
    return solutions, solved

def run_grover(oracle, backend, num_qubits, num_solutions):
    clbits = ClassicalRegister(num_qubits)
    qubits = QuantumRegister(num_qubits)

    n_iterations = np.pi/4 * np.sqrt((2**num_qubits) / np.round(num_solutions))
    grover = DistributedGroverAlgorithm.grover_operator_for_counting(int(np.floor(n_iterations)), oracle)

    qc = QuantumCircuit(qubits, clbits)
    qc.h(qubits)
    qc.append(grover, qubits)
    qc.measure(qubits, clbits)

    transpiled_qc = transpile(qc, backend)

    shots = 2048
    result = DistributedGroverAlgorithm.execute_on_the_backend(backend=backend, list_qc=[transpiled_qc], shots=shots, tags=["grover_experiment"]).result().get_counts() 

    counts = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    solutions = list(dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)).keys())[:num_solutions]
    print(solutions)
    print(counts)
    return solutions