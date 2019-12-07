#ifndef __CUDA_ERROR_CHECK__
#define __CUDA_ERROR_CHECK__

/**********************************************************************/
/* Code borrowed from PCL (https://github.com/PointCloudLibrary/pcl)  */
/**********************************************************************/

#include "cuda_runtime_api.h"
#include <stdio.h>
#include <stdlib.h>

#if defined(__GNUC__)
#define cudaErrorCheck(expr)  __errorCheck(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
#define cudaErrorCheck(expr)  __errorCheck(expr, __FILE__, __LINE__)    
#endif

static inline void __errorCheck(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
    {
        printf("[ERROR] %s \t %s:%d\n", cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

#endif
