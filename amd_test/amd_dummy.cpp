#include <hip/hip_runtime.h>
#include <math.h>
#include <rocblas/rocblas.h>
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

void sampleNormal(float *X, int size, float mean, float var)
{
    float x, y, z, std, val;
    for (int i = 0; i < size; i++)
    {
        x = (float)rand() / (float)RAND_MAX;
        y = (float)rand() / (float)RAND_MAX;
        z = sqrtf(-2 * logf(x)) * cosf(2 * M_PI * y);
        std = sqrtf(var);
        val = std * z + mean;
        X[i] = val;
    }
}

int main(int argc, char **argv)
{

    rocblas_status rstatus = rocblas_status_success;

    typedef float dataType;

    rocblas_int M = 8192;
    rocblas_int N = 8192 * 4;
    rocblas_int K = 8192;

    float hAlpha = 1;
    float hBeta = 0;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_int lda, ldb, ldc, sizeA, sizeB, sizeC;
    int strideA1, strideA2, strideB1, strideB2;

    if (transA == rocblas_operation_none)
    {
        lda = M;
        sizeA = K * lda;
        strideA1 = 1;
        strideA2 = lda;
    }
    else
    {
        lda = K;
        sizeA = M * lda;
        strideA1 = lda;
        strideA2 = 1;
    }
    if (transB == rocblas_operation_none)
    {
        ldb = K;
        sizeB = N * ldb;
        strideB1 = 1;
        strideB2 = ldb;
    }
    else
    {
        ldb = N;
        sizeB = K * ldb;
        strideB1 = ldb;
        strideB2 = 1;
    }

    ldc = M;
    sizeC = N * ldc;

    // using rocblas API
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    if (rstatus != rocblas_status_success)
    {
        fprintf(stderr, "Error creating ROCBLAS handle: %s\n", rocblas_status_to_string(rstatus));
        // Handle the error or exit the program
    }

    CHECK_ROCBLAS_STATUS(rstatus);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    float *hA, *hB, *hC;
    hA = (float *)malloc(sizeA * sizeof(float));
    hB = (float *)malloc(sizeB * sizeof(float));
    hC = (float *)malloc(sizeC * sizeof(float));

    sampleNormal(hA, sizeA, 0, 1);
    sampleNormal(hB, sizeB, 0, 1);

    // allocate memory on device
    float *dA, *dB, *dC;
    hipMalloc(&dA, sizeA * sizeof(float));
    hipMalloc(&dB, sizeB * sizeof(float));
    hipMalloc(&dC, sizeC * sizeof(float));

    if (!dA || !dB || !dC)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return EXIT_FAILURE;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(float) * sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(float) * sizeB, hipMemcpyHostToDevice));

    // enable passing alpha parameter from pointer to host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    // asynchronous calculation on device, returns before finished calculations
    rstatus = rocblas_sgemm(handle, transA, transB, M, N, K, &hAlpha, dA, lda, dB, ldb, &hBeta, dC, ldc);

    // check that calculation was launched correctly on device, not that result
    // was computed yet
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch device memory results, automatically blocked until results ready
    CHECK_HIP_ERROR(hipMemcpy(hC, dC, sizeof(float) * sizeC, hipMemcpyDeviceToHost));

    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    free(hA);
    free(hB);
    free(hC);

    return EXIT_SUCCESS;
}
