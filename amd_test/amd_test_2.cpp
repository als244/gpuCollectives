#include <rccl/rccl.h>
#include <time.h>

#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP_ERROR(error)                \
  if (error != hipSuccess)                    \
  {                                           \
    fprintf(stderr,                           \
            "HIP error: '%s'(%d) at %s:%d\n", \
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

  rcclComm_t comms[nDev];

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

  // initializing RCCL
  for (int i = 0; i < nDev; ++i)
  {
    RCCLCHECK(rcclCommInitRank(comms + i, nDev, 0, i));
  }

  // calling RCCL communication API. Group API is required when using
  // multiple devices per thread
  //  Starts the CUDA timer
  start = clock();
  RCCLCHECK(rcclGroupStart());
  for (int i = 0; i < nDev; ++i)
    RCCLCHECK(rcclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, rcclFloat, rcclSum, comms[i], s[i]));
  RCCLCHECK(rcclGroupEnd());

  // synchronizing on HIP streams to wait for completion of RCCL operation
  for (int i = 0; i < nDev; ++i)
  {
    CHECK_HIP_ERROR(hipSetDevice(i));
    CHECK_HIP_ERROR(hipStreamSynchronize(s[i]));
  }
  end = clock();

  // free device buffers
  for (int i = 0; i < nDev; ++i)
  {
    CHECK_HIP_ERROR(hipSetDevice(i));
    CHECK_HIP_ERROR(hipFree(sendbuff[i]));
    CHECK_HIP_ERROR(hipFree(recvbuff[i]));
  }

  milliseconds = (double)(end - start) / CLOCKS_PER_SEC;

  printf("%1.31f\n", milliseconds);

  // finalizing RCCL
  for (int i = 0; i < nDev; ++i)
    RCCLCHECK(rcclCommDestroy(comms[i]));

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
