# refer to https://github.com/reed-lau/cute-gemm/blob/main/profile.sh
# root user

ncu --csv --log-file a.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum python ./ld_mma/benchmarks/matmul_benchmark.py

python ./ld_mma/benchmarks/stat-csv.py