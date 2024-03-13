# ./ld_mma/tests/profile.sh
ncu -f --set full -o cublas_matmul python ./ld_mma/tests/cublas_matmul_test.py
ncu -f --set full -o warp_matmul python ./ld_mma/tests/warp_matmul_test.py
ncu -f --set full -o cute_matmul_v1 -k CuteMatmulV1 python ./ld_mma/tests/cute_matmul_v1_test.py
ncu -f --set full -o cute_matmul_v2 -k CuteMatmulV2 python ./ld_mma/tests/cute_matmul_v2_test.py
ncu -f --set full -o cute_matmul_v3 -k CuteMatmulV3 python ./ld_mma/tests/cute_matmul_v3_test.py
ncu -f --set full -o cute_matmul_v4 -k CuteMatmulV4 python ./ld_mma/tests/cute_matmul_v4_test.py