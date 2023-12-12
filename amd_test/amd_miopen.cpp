#include <miopen/miopen.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_MIOPEN_STATUS(status)                                             \
  if (status != miopenStatusSuccess)                                            \
  {                                                                             \
    fprintf(stderr, "MIOpen error: %d at %s:%d\n", status, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                                         \
  }

int main()
{
  miopenHandle_t miopenHandle;
  miopenStatus_t status = miopenCreate(&miopenHandle);

  CHECK_MIOPEN_STATUS(status);

  // Use the miopenHandle for MIOpen operations

  // ...

  // Close the MIOpen handle
  status = miopenDestroy(miopenHandle);
  CHECK_MIOPEN_STATUS(status);

  return 0;
}
