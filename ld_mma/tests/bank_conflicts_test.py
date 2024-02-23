# python ./ld_mma/tests/bank_conflicts_test.py
# nsys profile python ./ld_mma/tests/bank_conflicts_test.py
# ncu -f --set full -o cute_matmul_v3 python ./ld_mma/tests/bank_conflicts_test.py
# ncu --metrics  l1tex__data_bank_conflicts_pipe_lsu_mem_shared,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum python ./ld_mma/tests/bank_conflicts_test.py

import torch
import ld_mma
from ld_mma.ld_mma_interface import bank_conflicts_v1, bank_conflicts_v2, bank_conflicts_v3, bank_conflicts_v4
from ld_mma.ld_mma_interface import bank_conflicts_v5, bank_conflicts_v6

# bank_conflicts_v1(1)
# bank_conflicts_v2(1)
# bank_conflicts_v3(1)
# bank_conflicts_v4(1)
# bank_conflicts_v5(1)
bank_conflicts_v6(1)
print("end")