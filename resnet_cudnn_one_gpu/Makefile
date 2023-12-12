CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -Wall

all: ResNetFast ResNetFastOpt

ResNetFast: resnet_cudnn_fast.cu
	nvcc -g -G -arch=sm_80 resnet_cudnn_fast.cu --use_fast_math -lcurand -lcudnn -o ResNetFast

ResNetFastOpt: resnet_cudnn_fast.cu
	nvcc -O3 -arch=sm_80 resnet_cudnn_fast.cu --use_fast_math -lcurand -lcudnn -o  ResNetFastOpt


