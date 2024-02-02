import numpy as np
import qiskit.quantum_info as qi
from circuit import run_algo, run_counting, run_grover
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import GroverOperator, Diagonal

def convert(n_bits, solutions):
    exp = ""
    binary_list = []
    for item in solutions:
        bin_str = np.binary_repr(item, n_bits)
        binary_list.append(bin_str)
        logic_str = ""
        for i in range(len(bin_str)):
            to_add = ""
            if bin_str[i] == '0':
                to_add = "~b" + str(i)
            else:
                to_add = "b" + str(i)

            if len(logic_str) == 0:
                logic_str += to_add
            else:
                logic_str += " & " + to_add

        if len(exp) == 0:
            exp += "(" + logic_str + ")"
        else:
            exp += " | (" + logic_str + ")"

    return exp, binary_list


tag = "test"
token="d9953c32517b3d39b0d38dc846969b744068c3fd6ca8e1b67c6359d4d6eabfb5626be32f99903cfdcec3385d389d770c44ade0bb354b4b5692e7882a142d458b"

service = QiskitRuntimeService(channel="ibm_quantum", token=token)
backend = service.get_backend('ibmq_qasm_simulator')


expression = '~(q0 & q1 & q2 & q3 & q4 & q5 & q6)'
# 1110 -> 0111
# 1100 -> 0011
# 0011 -> 1100
# 1011 -> 1101

test_cases = []
'''
number_of_tests = 30
n = 7
k = 2
def generate_tests(n, k, number_of_tests, test_cases):
    number_of_answers = np.random.randint(1, high=2**(n-1) + 1, size=number_of_tests)
    for i in range(number_of_tests):
        arr = [1 for i in range(2**n)]
        expected_solutions = []
        answers_idx = np.random.randint(0, high=2**n, size=number_of_answers[i])
        for j in range(number_of_answers[i]):
            while arr[answers_idx[j]] == -1:
                answers_idx[j] = np.random.randint(0, high=2**n, size=1)[0]
            arr[answers_idx[j]] = -1
            expected_solutions.append(np.binary_repr(answers_idx[j], n))
        oracle = Diagonal(arr) #8/8 solutions
        test_cases.append((oracle, n, k, expected_solutions))

generate_tests(5, 3, 30, test_cases)
generate_tests(6, 3, 30, test_cases)
generate_tests(7, 3, 30, test_cases)

global_distributed_tp = 0
global_distributed_fp = 0
global_distributed_fn = 0


global_grover_tp = 0
global_grover_fp = 0
global_grover_fn = 0

distributed_tp_l = []
distributed_fp_l = []
distributed_fn_l = []

grover_tp_l = []
grover_fp_l = []
grover_fn_l = []

expected_l = []

global_total_distributed_found = 0
global_total_grover_found = 0
global_total_expected = 0

import sys
from datetime import datetime
np.set_printoptions(threshold=sys.maxsize)
# print(np.array(qi.Operator(oracle).data))

now = datetime.now()   
orig_stdout = sys.stdout
f = open(f'distributed_out_{now.time()}.txt', 'w')
sys.stdout = f


for i, (oracle, n_qubits, k, expected_solutions_t) in enumerate(test_cases):
    global_total_expected += len(expected_solutions_t)
    expected_l.append(len(expected_solutions_t))
    print("number of solutions expected: ", len(expected_solutions_t))
    answers_grover, answers_splitting = run_algo(oracle, backend, n_qubits, k, f"test_circuit_{n_qubits}_{k}_expression_id_{i}_{tag}")
    all_answers = set([item for item in answers_grover]).union(set([item for item in answers_splitting]))
    print("found", all_answers)
    print("expected", expected_solutions_t)
    expected_solutions = set(expected_solutions_t)
    distributed_fp = len(all_answers.difference(expected_solutions))
    distributed_fn = len(expected_solutions.difference(all_answers))
    distributed_tp = len(all_answers) - distributed_fp

    global_distributed_tp += distributed_tp
    global_distributed_fp += distributed_fp
    global_distributed_fn += distributed_fn
    global_total_distributed_found += len(all_answers)
    print(f"DISTRIBUTED_METRICS_TEST:{distributed_tp},{distributed_fp},{distributed_fn},{len(all_answers)}")
    print(f"GLOBAL_METRICS_DISTRIBUTED:{global_distributed_tp},{global_distributed_fp},{global_distributed_fn},{global_total_distributed_found}")
    if all_answers.issubset(expected_solutions) and expected_solutions.issubset(all_answers):
        print(f"DISTRIBUTED Correct answer for test {i}")
    else:
        print(f"!!!!DISTRIBUTED Incorrect answer for test {i}")
        print(f"In all_answers but not in expected_solutions {all_answers.difference(expected_solutions)}")
        print(f"In expected_solutions but not in all_answers {expected_solutions.difference(all_answers)}")

    print("TESTCASE ", i, " SOLUTIONS: ", expected_solutions_t, "NUM OF SOLUTIONS: ", len(expected_solutions_t))
    solutions_found = run_grover(oracle, backend, n_qubits, len(expected_solutions_t))
    expected_solutions = set(expected_solutions_t)
    solutions_found = set(solutions_found)
    print("found", solutions_found)
    print("expected", expected_solutions)
    grover_fp = len(solutions_found.difference(expected_solutions))
    grover_fn = len(expected_solutions.difference(solutions_found))
    grover_tp = len(solutions_found) - grover_fp

    global_grover_tp += grover_tp
    global_grover_fp += grover_fp
    global_grover_fn += grover_fn
    global_total_grover_found += len(solutions_found)
    print(f"GROVER_METRICS_TEST:{grover_tp},{grover_fp},{grover_fn},{len(solutions_found)}")
    print(f"GLOBAL_METRICS_GROVER:{global_grover_tp},{global_grover_fp},{global_grover_fn},{global_total_grover_found}")
    print(f"GLOBAL_EXPECTED:{global_total_expected}")
    if solutions_found.issubset(expected_solutions) and solutions_found.issubset(expected_solutions):
        print(f"GROVER Correct answer for the testcase")
    else:
        print(f"!!!!GROVER Incorrect answer for testcase")
        print(f"In all_answers but not in expected_solutions {solutions_found.difference(expected_solutions)}")
        print(f"In expected_solutions but not in all_answers {expected_solutions.difference(solutions_found)}")

    distributed_tp_l.append(distributed_tp)
    distributed_fp_l.append(distributed_fp)
    distributed_fn_l.append(distributed_fn)

    grover_tp_l.append(grover_tp)
    grover_fp_l.append(grover_fp)
    grover_fn_l.append(grover_fn)
print(f"GLOBAL_METRICS_GROVER:{global_grover_tp},{global_grover_fp},{global_grover_fn}")
print(f"GLOBAL_METRICS_DISTRIBUTED:{global_distributed_tp},{global_distributed_fp},{global_distributed_fn}")
print("EXPECTED_N", expected_l)
print("distributed_tp_l", distributed_tp_l)
print("distributed_fp_l", distributed_fp_l)
print("distributed_fn_l", distributed_fn_l)
print("grover_tp_l", grover_tp_l)
print("grover_fp_l", grover_fp_l)
print("grover_fn_l", grover_fn_l)
f.close()
'''

import sys
from datetime import datetime

now = datetime.now()   
orig_stdout = sys.stdout
f = open(f'out_{now.time()}.txt', 'w')
sys.stdout = f

res_our = 0
res_qiskit = 0

for j in range(2**7 + 1):
    arr = [1 for i in range(2**7)]
    for i in range(j):
        arr[i] = -1
    oracle = Diagonal(arr) #8/8 solutions
    print(np.array(qi.Operator(oracle).data))
    answer = j
    print("EXPECT ANSWER: ", answer)
    our, qiksit = run_counting(oracle, backend, 7, int(np.floor(7/2)), f"(n-k)/2_{answer}")
    if our == answer:
        res_our += 1
    if qiksit == answer:
        res_qiskit += 1
    else:
        print("QISKIT FAILED AT ", j)
    print(f"OUR SCORE INTER:{res_our}/{2**7}")
    print(f"QISKIT SCORE INTER:{res_qiskit}/{2**7}")

print(f"OUR SCORE:{res_our}/{2**7}")
print(f"QISKIT SCORE:{res_qiskit}/{2**7}")

sys.stdout = orig_stdout

f.close()
