# python ./ld_mma/tests/bank_conflicts_test.py
# nsys profile python ./ld_mma/tests/bank_conflicts_test.py
# ncu -f --set full -o cute_matmul_v3 python ./ld_mma/tests/bank_conflicts_test.py
# ncu --metrics  l1tex__data_bank_conflicts_pipe_lsu_mem_shared,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum python ./ld_mma/tests/bank_conflicts_test.py

import torch
import ld_mma
from ld_mma.ld_mma_interface import bank_conflicts

bank_conflicts(1)
print("end")