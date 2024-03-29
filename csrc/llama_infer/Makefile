# CC_FILES=$(shell find ./ -name "*.cu")
CC_FILES=./run_2.cu ./run_1.cu
EXE_FILES=$(CC_FILES:.cu=)
# -arch=sm_86 
all:$(EXE_FILES)

%:%.cu
	nvcc -o $@ $< -arch=sm_86 -O3 -std=c++17 -I3rd/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas

clean:
	rm -rf $(EXE_FILES)
	