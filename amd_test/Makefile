CC = hipcc
CFLAGS = -g3 -std=c99 -pedantic -Wall

all: getDevices amdTest

getDevices: amd_get_devices.cpp
	hipcc amd_get_devices.cpp -o getDevices

amdTest: amd_dummy.cpp
	hipcc amd_dummy.cpp -lrocblas -o amdTest -lm

miOpen: amd_miopen.cpp
	hipcc amd_dummy.cpp -lhip -o miOpen -lmiopen
