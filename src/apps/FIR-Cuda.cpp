//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf
{
namespace apps
{

#define USE_CUDA_CONSTANT_MEMORY
//#undef USE_CUDA_CONSTANT_MEMORY

#if defined(USE_CUDA_CONSTANT_MEMORY)

__constant__ Real_type coeff[FIR_COEFFLEN];

#define FIR_DATA_SETUP_CUDA \
  Real_type *dcoeff_addr; \
  CAMP_CUDA_API_INVOKE_AND_CHECK( cudaGetSymbolAddress, \
      (void**)&dcoeff_addr, coeff ); \
  CAMP_CUDA_API_INVOKE_AND_CHECK( cudaMemcpyAsync, \
      dcoeff_addr, coeff_array, FIR_COEFFLEN * sizeof(Real_type), \
      cudaMemcpyHostToDevice, res.get_stream() );


#define FIR_DATA_TEARDOWN_CUDA

// Original kernel (kept for RAJA_CUDA variant which uses FIR_BODY macro)
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void fir(Real_ptr out, Real_ptr in,
                    const Index_type coefflen,
                    Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     FIR_BODY;
   }
}

// Optimized kernel: shared memory tiling to reduce redundant global loads.
// Each thread in the original reads FIR_COEFFLEN consecutive doubles from
// global memory; adjacent threads overlap by FIR_COEFFLEN-1 elements.
// This version cooperatively loads the tile into shared memory, then each
// thread reads from fast shared memory instead.
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void fir_opt(Real_type* __restrict__ out,
                        const Real_type* __restrict__ in,
                        const Index_type coefflen,
                        Index_type iend)
{
   __shared__ Real_type s_in[block_size + FIR_COEFFLEN - 1];

   const Index_type base = blockIdx.x * block_size;
   const Index_type tid = threadIdx.x;
   constexpr Index_type tile_size = block_size + FIR_COEFFLEN - 1;

   // Cooperative load: each thread loads ~1 element, extras load the tail
   for (Index_type k = tid; k < tile_size; k += block_size) {
     Index_type gi = base + k;
     s_in[k] = (gi < iend + FIR_COEFFLEN - 1) ? in[gi] : 0.0;
   }
   __syncthreads();

   Index_type i = base + tid;
   if (i < iend) {
     Real_type sum = 0.0;
     #pragma unroll
     for (Index_type j = 0; j < FIR_COEFFLEN; j++) {
       sum += coeff[j] * s_in[tid + j];
     }
     out[i] = sum;
   }
}

#else  // use global memory for coefficients

#define FIR_DATA_SETUP_CUDA \
  Real_ptr coeff; \
  \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocData(DataSpace::CudaDevice, coeff, FIR_COEFFLEN); \
  copyData(DataSpace::CudaDevice, coeff, DataSpace::Host, tcoeff, FIR_COEFFLEN);


#define FIR_DATA_TEARDOWN_CUDA \
  deallocData(DataSpace::CudaDevice, coeff);

// Original kernel (kept for RAJA_CUDA variant which uses FIR_BODY macro)
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void fir(Real_ptr out, Real_ptr in,
                    Real_ptr coeff,
                    const Index_type coefflen,
                    Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     FIR_BODY;
   }
}

// Optimized kernel: shared memory tiling (non-constant-memory variant)
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void fir_opt(Real_type* __restrict__ out,
                        const Real_type* __restrict__ in,
                        const Real_type* __restrict__ coeff,
                        const Index_type coefflen,
                        Index_type iend)
{
   __shared__ Real_type s_in[block_size + FIR_COEFFLEN - 1];

   const Index_type base = blockIdx.x * block_size;
   const Index_type tid = threadIdx.x;
   constexpr Index_type tile_size = block_size + FIR_COEFFLEN - 1;

   // Cooperative load: each thread loads ~1 element, extras load the tail
   for (Index_type k = tid; k < tile_size; k += block_size) {
     Index_type gi = base + k;
     s_in[k] = (gi < iend + FIR_COEFFLEN - 1) ? in[gi] : 0.0;
   }
   __syncthreads();

   Index_type i = base + tid;
   if (i < iend) {
     Real_type sum = 0.0;
     #pragma unroll
     for (Index_type j = 0; j < FIR_COEFFLEN; j++) {
       sum += coeff[j] * s_in[tid + j];
     }
     out[i] = sum;
   }
}

#endif


template < size_t block_size >
void FIR::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  FIR_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    FIR_COEFF;

    FIR_DATA_SETUP_CUDA;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

#if defined(USE_CUDA_CONSTANT_MEMORY)
      const Real_type* cin = in;

      RPlaunchCudaKernel( (fir_opt<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          out, cin,
                          coefflen,
                          iend );
#else
      const Real_type* cin = in;
      const Real_type* ccoeff = coeff;

      RPlaunchCudaKernel( (fir_opt<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          out, cin,
                          ccoeff,
                          coefflen,
                          iend );
#endif

    }
    stopTimer();

    FIR_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    FIR_COEFF;

    FIR_DATA_SETUP_CUDA;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIR_BODY;
       });

    }
    stopTimer();

    FIR_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  FIR : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(FIR, Cuda, Base_CUDA, RAJA_CUDA)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
