# refer to https://github.com/reed-lau/cute-gemm/blob/main/profile.sh
# root user

ncu --csv --log-file a.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum python ./ld_mma/benchmarks/matmul_benchmark.py
ncu --csv --log-file b.csv --cache-control=all --clock-control=base --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum python ./ld_mma/benchmarks/matmul_benchmark.py
ncu --csv --log-file b.csv --cache-control=all --clock-control=base --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum python ./ld_mma/benchmarks/matmul_benchmark.py
ncu   --csv --log-file b.csv --metrics  l1tex__data_bank_conflicts_pipe_lsu_mem_shared,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum python ./ld_mma/benchmarks/matmul_benchmark.py

sm__warps_active.avg.pct_of_peak_sustained_active
python ./ld_mma/benchmarks/stat-csv.py