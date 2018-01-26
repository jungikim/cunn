#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "common.h"

template <typename T, typename AccumT>
__global__ void cunn_SVDLinear_updateFullView_kernel(
    const int nthreads,
    long *indices,
    T *z,
    T *B,
    T *h,
    T *bias,
    const int N,
    const int batchSize,
    const int V,
    const int D)
{
  CUDA_KERNEL_LOOP(index, nthreads) {

    const int nIdx = index / batchSize;
    const int bIdx = index % batchSize;

    long vIdx = indices[nIdx * batchSize + bIdx] - 1;
    AccumT dot = AccumT(0);
    for(int dIdx = 0 ; dIdx < D ; dIdx ++)
        dot += THCNumerics<T>::mul(B[vIdx * D + dIdx],
                                   h[dIdx * batchSize + bIdx]);

    if (bias)
      dot += bias[vIdx];

    z[vIdx * batchSize + bIdx] = ScalarConvert<AccumT, T>::to(dot);

    index += blockDim.x;
  }
}

#include "generic/SVDLinear.cu"
#include "THCGenerateHalfType.h"
#include "generic/SVDLinear.cu"
#include "THCGenerateFloatType.h"
#include "generic/SVDLinear.cu"
#include "THCGenerateDoubleType.h"
