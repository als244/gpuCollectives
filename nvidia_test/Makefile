CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -Wall
PROGNAME = SingProcMultDev*

all: single_proc_multi_dev_allReduce single_proc_multi_dev_Broadcast single_proc_multi_dev_Reduce one_dev_per_thr

single_proc_multi_dev_allReduce: single_process_multi_device_allReduce.cu
	nvcc -I /usr/include -L /lib64 -lnccl -g -G -arch=sm_80 single_process_multi_device_allReduce.cu -o SingProcMultDevAllReduce

single_proc_multi_dev_Broadcast: single_process_multi_device_Broadcast.cu
	nvcc -I /usr/include -L /lib64 -lnccl -g -G -arch=sm_80 single_process_multi_device_Broadcast.cu -o SingProcMultDevBroadcast

single_proc_multi_dev_Reduce: single_process_multi_device_Reduce.cu
	nvcc -I /usr/include -L /lib64 -lnccl -g -G -arch=sm_80 single_process_multi_device_Reduce.cu -o SingProcMultDevReduce

one_dev_per_thr: one_device_per_thread.cu
	nvcc -I /usr/include -L /lib64 -lnccl -lmpi -g -G -arch=sm_80  one_device_per_thread.cu -o OneDevPerThread

clean:
	rm slurm-*.out rm $(PROGNAME)