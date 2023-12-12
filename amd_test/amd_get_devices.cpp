#include <hip/hip_runtime.h>
// #include <hip_runtime_api.h>
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

#define CHECK_ROCBLAS_STATUS(status)                  \
    if (status != rocblas_status_success)             \
    {                                                 \
        fprintf(stderr, "rocBLAS error: ");           \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }

int main(int argc, char **argv)
{

    int ndevs = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&ndevs));

    devProp = hipGetDeviceProperties();
    for (int i = 0; i < ndevs; i++)
    {
        CHECK_HIP_ERROR(hipGetDeviceCount(&devProp, i));
        printf("Device %i:\n\tName: %s\n\tMulti-Processor Count: %d\n", i, devProp.name, devProp.multiProcessorCount);
    }
}
