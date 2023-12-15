// The following code depicts a complete working example
// with a single process that manages multiple devices:
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "/usr/include/nccl.h"
#include <time.h>

#define CUDACHECK(cmd)                                     \
  do                                                       \
  {                                                        \
    cudaError_t err = cmd;                                 \
    if (err != cudaSuccess)                                \
    {                                                      \
      printf("Failed: Cuda error %s:%d '%s'\n",            \
             __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                  \
    }                                                      \
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
  fprintf(file, "%ld,", size);
  fprintf(file, "%i", nDev);
  fprintf(file, "\n");

  fclose(file);

  printf("Success \n");
  return 0;
}
