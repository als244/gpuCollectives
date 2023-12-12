#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP_ERROR(error)                    \
    if (error != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }

int main(int argc, char **argv) {


	int ndev = 0;
	CHECK_HIP_ERROR(hipGetDeviceCount(&ndev));

	hipDeviceProp_t devProp;
	for (int i = 0; i < ndev; i++){
		CHECK_HIP_ERROR(hipGetDeviceProperties(&devProp, i));
		printf("Device #%d:\n\tName: %s\n\tMulti-Processor Count: %d\n", i, devProp.name, devProp.multiProcessorCount);
	}

	return 0;
}

