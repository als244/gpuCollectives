// The following code depicts a complete working example
// with a single process that manages multiple devices:
// #include "/opt/rccl/include/rccl.h"
#include <rccl/rccl.h>
#include <time.h>

#include <hip/hip_runtime.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP_ERROR(error)                \
  if (error != hipSuccess)                    \
  {                                           \
    fprintf(stderr,                           \
            "hip error: '%s'(%d) at %s:%d\n", \
            hipGetErrorString(error),         \
            error,                            \
            __FILE__,                         \
            __LINE__);                        \
    exit(EXIT_FAILURE);                       \
  }

#define CHECK_ROCBLAS_STATUS(status)              \
  if (status != rocblas_status_success)           \
  {                                               \
    fprintf(stderr, "rocBLAS error: ");           \
    fprintf(stderr,                               \
            "rocBLAS error: '%s'(%d) at %s:%d\n", \
            rocblas_status_to_string(status),     \
            status,                               \
            __FILE__,                             \
            __LINE__);                            \
    exit(EXIT_FAILURE);                           \
  }

#define RCCLCHECK(cmd)                                     \
  do                                                       \
  {                                                        \
    rcclResult_t res = cmd;                                \
    if (res != rcclSuccess)                                \
    {                                                      \
      printf("Failed, RCCL error %s:%d '%s'\n",            \
             __FILE__, __LINE__, rcclGetErrorString(res)); \
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
  int size = atoi(argv[2]);
  ncclComm_t comms[nDev];
  // managing 2 devices
  //  int nDev = 2;
  //  int size = 32*1024*1024;
  int devs[nDev]; // = {0, 1}; //, 2, 3 };
  for (int i = 0; i < nDev; i++)
  {
    devs[i] = i;
  }
  double milliseconds;
  clock_t start, end;

  // allocating and initializing device buffers
  float **sendbuff = (float **)malloc(nDev * sizeof(float *));
  float **recvbuff = (float **)malloc(nDev * sizeof(float *));
  cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

  for (int i = 0; i < nDev; ++i)
  {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s + i));
  }

  // initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  // calling NCCL communication API. Group API is required when using
  // multiple devices per thread
  //  Starts the CUDA timer
  start = clock();
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat, ncclSum,
                            comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());

  // synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i)
  {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }
  end = clock();

  // free device buffers
  for (int i = 0; i < nDev; ++i)
  {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  milliseconds = (double)(end - start) / CLOCKS_PER_SEC;

  printf("%1.31f\n", milliseconds);

  // finalizing NCCL
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
  fprintf(file, "%i,", size);
  fprintf(file, "\n");

  fclose(file);

  printf("Success \n");
  return 0;
}
