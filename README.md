# LD_mma
Depend on pytorch and cuda

# compile
```
git clone --recursive https://github.com/l1351868270/LD_mma.git
or
git clone https://github.com/l1351868270/LD_mma.git
git submodule update --init --recursive
cd mma
python setup.py install

```
# 优化思路
## GPU上的Profiling
GPU上的Profiling分为两类:
1. 对系统整体（CPU&GPU）执行情况进行Profile,判断性能瓶颈是位于CPU还是GPU上,并考虑CPU&GPU之间的同步开销
2. 对GPU Kernel进行Profile,以找到Kernel的潜在优化点
# Profiling Tools
## CPU Profiler
### gprof
```
g++ -pg
-pg: profiler gprof
-O2 -O3 会inline函数

gprof 

```

### Intel VTune
### AMD uProf

## GPU Profiler
### NVIDIA Nsight

### NVIDIA Nsight Compute

# Matrials
## Video
[GPU编程](https://www.bilibili.com/video/BV1424y1i7xe)

[CUDA: From Correctness to Performance](https://wiki.lcpu.dev/hpc/from-scratch/cuda)


