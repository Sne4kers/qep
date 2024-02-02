import numpy as np

GLOBAL_EXPECTED = 278
GLOBAL_METRICS_GROVER_5 = (253,39,39)
GLOBAL_METRICS_DISTRIBUTED_5 = (292,0,0)
GLOBAL_METRICS_GROVER_6 = (772,69,69)
GLOBAL_METRICS_DISTRIBUTED_6 = (841,0,0)
GLOBAL_METRICS_GROVER_7 = (1724,186,186)
GLOBAL_METRICS_DISTRIBUTED_7 = (1868,54,42)

#GLOBAL_METRICS_GROVER = (GLOBAL_METRICS_GROVER_6[0] - GLOBAL_METRICS_GROVER_5[0],GLOBAL_METRICS_GROVER_6[1] - GLOBAL_METRICS_GROVER_5[1],GLOBAL_METRICS_GROVER_6[2] - GLOBAL_METRICS_GROVER_5[2])
#GLOBAL_METRICS_DISTRIBUTED = (GLOBAL_METRICS_DISTRIBUTED_6[0] - GLOBAL_METRICS_DISTRIBUTED_5[0],GLOBAL_METRICS_DISTRIBUTED_6[1] - GLOBAL_METRICS_DISTRIBUTED_5[1],GLOBAL_METRICS_DISTRIBUTED_6[2] - GLOBAL_METRICS_DISTRIBUTED_5[2])

GLOBAL_METRICS_GROVER = (GLOBAL_METRICS_GROVER_7[0] - GLOBAL_METRICS_GROVER_6[0] - GLOBAL_METRICS_GROVER_5[0],GLOBAL_METRICS_GROVER_7[1] - GLOBAL_METRICS_GROVER_6[1] - GLOBAL_METRICS_GROVER_5[1],GLOBAL_METRICS_GROVER_7[2] - GLOBAL_METRICS_GROVER_6[2] - GLOBAL_METRICS_GROVER_5[2])
GLOBAL_METRICS_DISTRIBUTED = (GLOBAL_METRICS_DISTRIBUTED_7[0] - GLOBAL_METRICS_DISTRIBUTED_6[0] - GLOBAL_METRICS_DISTRIBUTED_5[0],GLOBAL_METRICS_DISTRIBUTED_7[1] - GLOBAL_METRICS_DISTRIBUTED_6[1] - GLOBAL_METRICS_DISTRIBUTED_5[1],GLOBAL_METRICS_DISTRIBUTED_7[2] - GLOBAL_METRICS_DISTRIBUTED_6[2] - GLOBAL_METRICS_DISTRIBUTED_5[2])

EXPECTED_N = [8, 10, 24, 32, 19, 10, 4, 3, 8, 3, 22, 8, 6, 32, 9, 12, 29, 16, 8, 24, 5, 2, 24, 16, 8, 11, 21, 7, 23, 17]
distributed_tp_l = [9, 13, 15, 13, 15, 15, 16, 7, 1, 6, 15, 9, 12, 1, 3, 6, 4, 15, 15, 9, 11, 1, 11, 8, 2, 6, 2, 2, 14, 6]
distributed_fp_l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0]
distributed_fn_l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0]
grover_tp_l = [9, 13, 14, 13, 15, 15, 11, 7, 1, 15, 14, 9, 12, 1, 3, 6, 4, 14, 14, 9, 11, 1, 11, 15, 2, 6, 2, 2, 14, 6]
grover_fp_l = [0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
grover_fn_l = [0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


precision_distributed = GLOBAL_METRICS_DISTRIBUTED[0] / (GLOBAL_METRICS_DISTRIBUTED[0] + GLOBAL_METRICS_DISTRIBUTED[1])
recall_distributed = GLOBAL_METRICS_DISTRIBUTED[0] / (GLOBAL_METRICS_DISTRIBUTED[0] + GLOBAL_METRICS_DISTRIBUTED[2])
F1_score_distributed = 2 * precision_distributed * recall_distributed / (precision_distributed + recall_distributed)


print("precision distributed: ", precision_distributed)
print("recall distributed: ", recall_distributed)
print("f1 distributed: ", F1_score_distributed)

precision_grover = GLOBAL_METRICS_GROVER[0] / (GLOBAL_METRICS_GROVER[0] + GLOBAL_METRICS_GROVER[1])
recall_grover = GLOBAL_METRICS_GROVER[0] / (GLOBAL_METRICS_GROVER[0] + GLOBAL_METRICS_GROVER[2])
F1_score_grover = 2 * precision_grover * recall_grover / (precision_grover + recall_grover)


print("precision grover: ", precision_grover)
print("recall grover: ", recall_grover)
print("f1 grover: ", F1_score_grover)

average_proportion = np.mean(EXPECTED_N) / 2**6
print("average_proportion: ", average_proportion)