#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#define HIPCHECK(cmd)                                        \
  do                                                         \
  {                                                          \
    hipError_t error = cmd;                                  \
    if (error != hipSuccess)                                 \
    {                                                        \
      fprintf(stderr, "Failed, HIP error %s:%d '%s'\n",      \
              __FILE__, __LINE__, hipGetErrorString(error)); \
      return EXIT_FAILURE;                                   \
    }                                                        \
  } while (0)

int main()
{
  int nDev = 2;    // Set the number of devices
  int size = 1000; // Set the size of the data array

  // Allocating and initializing device buffers
  float **sendbuff = (float **)malloc(nDev * sizeof(float *));
  float **recvbuff = (float **)malloc(nDev * sizeof(float *));
  hipStream_t *s = (hipStream_t *)malloc(sizeof(hipStream_t) * nDev);

  for (int i = 0; i < nDev; ++i)
  {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMalloc(sendbuff + i, size * sizeof(float)));
    HIPCHECK(hipMalloc(recvbuff + i, size * sizeof(float)));
    HIPCHECK(hipMemset(sendbuff[i], 1, size * sizeof(float)));
    HIPCHECK(hipMemset(recvbuff[i], 0, size * sizeof(float)));
    HIPCHECK(hipStreamCreate(s + i));
  }

  // kernel launch
  for (int i = 0; i < nDev; ++i)
  {
    HIPCHECK(hipSetDevice(i));
    hipLaunchKernelGGL((kernel_function), dim3(blocks), dim3(threads), 0, s[i], sendbuff[i], recvbuff[i], size);
  }

  // Synchronizing on HIP streams to wait for completion of the operations
  for (int i = 0; i < nDev; ++i)
  {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamSynchronize(s[i]));
  }

  // Free device buffers
  for (int i = 0; i < nDev; ++i)
  {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipFree(sendbuff[i]));
    HIPCHECK(hipFree(recvbuff[i]));
  }

  // Finalizing HIP streams
  for (int i = 0; i < nDev; ++i)
    HIPCHECK(hipStreamDestroy(s[i]));

  return 0;
}
