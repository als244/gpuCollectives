CC = hipcc
CFLAGS = -g3 -std=c99 -pedantic -Wall

all: getDevices amdTest miOpen spmd

getDevices: amd_get_devices.cpp
	hipcc amd_get_devices.cpp -o getDevices

amdTest: amd_dummy.cpp
	hipcc amd_dummy.cpp -lrocblas -o amdTest -lm

miOpen: amd_miopen.cpp
	hipcc amd_miopen.cpp -o miOpen -lm -L/opt/rocm/miopen/lib -lMIOpen

# rccl_test: amd_test.cpp
# 	hipcc amd_test.cpp -o miOpen -lm -lrccl

spmd: single_process_multi_device_allReduce.cpp
	hipcc single_process_multi_device_allReduce.cpp -o spmd -lm -lrccl

# nvcc -I /usr/include -L /lib64 -lnccl -g -G -arch=sm_80 single_process_multi_device_allReduce.cu -o SingProcMultDevAllReduce
