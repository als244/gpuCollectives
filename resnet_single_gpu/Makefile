CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -Wall

all: ResNetCuDNN ResNetCuDNNOpt ResNetMIOpen ResNetMIOpenOpt

ResNetCuDNN: resnet_cudnn_fast.cu
	nvcc -g -G -arch=sm_80 resnet_cudnn_fast.cu --use_fast_math -lcurand -lcudnn -o ResNetCuDNN

ResNetCuDNNOpt: resnet_cudnn_fast.cu
	nvcc -O3 -arch=sm_80 resnet_cudnn_fast.cu --use_fast_math -lcurand -lcudnn -o  ResNetCuDNNOpt

ResNetMIOpen: resnet_miopen_fast.cpp
	hipcc -g --offload-arch=gfx1100 resnet_miopen_fast.cpp -ffast-math -lMIOpen -lhiprand -o ResNetMIOpen

ResNetMIOpenOpt: resnet_miopen_fast.cpp
	hipcc -O3 --offload-arch=gfx1100 resnet_miopen_fast.cpp -ffast-math -lMIOpen -lhiprand -o ResNetMIOpenOpt


