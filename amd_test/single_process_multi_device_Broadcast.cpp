// The following code depicts a complete working example
// with a single process that manages multiple devices:
#include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include "/usr/include/nccl.h"
#include <time.h>

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP_ERROR(error)                            \
  do                                                      \
  {                                                       \
    hipError_t err = error;                               \
    if (err != hipSuccess)                                \
    {                                                     \
      printf("Failed: HIP error %s:%d '%s'\n",            \
             __FILE__, __LINE__, hipGetErrorString(err)); \
      exit(EXIT_FAILURE);                                 \
    }                                                     \
  } while (0)

#define NCCLCHECK(cmd)                                     \
  do                                                       \
  {                                                        \
    ncclResult_t res = cmd;                                \
    if (res != ncclSuccess)                                \
    {                                                      \
      printf("Failed, NCCL error %s:%d '%s'\n",            \
             __FILE__, __LINE__, ncclGetErrorString(res)); \
      exit(EXIT_FAILURE);                                  \
    }                                                      \
  } while (0)

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    fprintf(stderr, "Usage: %s <Number of GPUs> <Buffer Size> \n", argv[0]);
    exit(EXIT_FAILURE);
  }
  int nDev = atoi(argv[1]);
  long size = atol(argv[2]);

  ncclComm_t comms[nDev];
  int devs[nDev];
  for (int i = 0; i < nDev; i++)
  {
    devs[i] = i;
  }
  double milliseconds;
  clock_t start, end;

  // Allocating and initializing device buffers
  float **sendbuff = (float **)malloc(nDev * sizeof(float *));
  float **recvbuff = (float **)malloc(nDev * sizeof(float *));
  hipStream_t *s = (hipStream_t *)malloc(sizeof(hipStream_t) * nDev);

  for (int i = 0; i < nDev; ++i)
  {
    CHECK_HIP_ERROR(hipSetDevice(i));
    CHECK_HIP_ERROR(hipMalloc(sendbuff + i, size * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(recvbuff + i, size * sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(sendbuff[i], 1, size * sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(recvbuff[i], 0, size * sizeof(float)));
    CHECK_HIP_ERROR(hipStreamCreate(s + i));
  }

  // Initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  // Calling NCCL communication API. Group API is required when using
  // multiple devices per thread
  // Starts the HIP timer
  start = clock();
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclBroadcast((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat, devs[0],
                            comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());

  // Synchronizing on HIP streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i)
  {
    CHECK_HIP_ERROR(hipSetDevice(i));
    CHECK_HIP_ERROR(hipStreamSynchronize(s[i]));
  }
  end = clock();

  // Free device buffers
  for (int i = 0; i < nDev; ++i)
  {
    CHECK_HIP_ERROR(hipSetDevice(i));
    CHECK_HIP_ERROR(hipFree(sendbuff[i]));
    CHECK_HIP_ERROR(hipFree(recvbuff[i]));
  }

  milliseconds = (double)(end - start) / CLOCKS_PER_SEC;

  printf("%1.31f\n", milliseconds);

  // Finalizing NCCL
  for (int i = 0; i < nDev; ++i)
    ncclCommDestroy(comms[i]);

  // Dump to CSV
  FILE *file = fopen("output.csv", "a");
  if (file == NULL)
  {
    fprintf(stderr, "Error opening file!\n");
    exit(EXIT_FAILURE);
  }
  fprintf(file, "%s,", argv[0]);
  fprintf(file, "%1.31f,", milliseconds);
  fprintf(file, "%ld,", size);
  fprintf(file, "%i", nDev);
  fprintf(file, "\n");

  fclose(file);

  printf("Success \n");
  return 0;
}
