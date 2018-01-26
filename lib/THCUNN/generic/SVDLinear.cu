#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SVDLinear.cu"
#else

void THNN_(SVDLinear_updateFullView)(
           THCState *state,
           THCIndexTensor *indices,
           THCTensor *z,
           THCTensor *B,
           THCTensor *h,
           THCTensor *bias)
{
  THCUNN_assertSameGPU(state, 5, indices, z, B, h, bias);

  indices = THCIndexTensor_(newContiguous)(state, indices);
  z = THCTensor_(newContiguous)(state, z);
  B = THCTensor_(newContiguous)(state, B);
  h = THCTensor_(newContiguous)(state, h);
  if (bias)
    bias = THCTensor_(newContiguous)(state, bias);

  long N = THCIndexTensor_(size)(state, indices, 0);
  long V = THCTensor_(size)(state, z, 0);
  long D = THCTensor_(size)(state, h, 0);

  int count = THCIndexTensor_(nElement)(state, indices);

  long batchSize = 1;
  if (THCIndexTensor_(nDimension)(state, indices) > 1)
    batchSize = THCIndexTensor_(size)(state, indices, 1);

  cunn_SVDLinear_updateFullView_kernel<real, accreal>
    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count,
      THCIndexTensor_(data)(state, indices),
      THCTensor_(data)(state, z),
      THCTensor_(data)(state, B),
      THCTensor_(data)(state, h),
      THCTensor_(data)(state, bias),
      N,
      batchSize,
      V,
      D
  );
}

#endif
