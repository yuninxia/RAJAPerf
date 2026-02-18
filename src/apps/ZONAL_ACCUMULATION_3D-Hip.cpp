//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ZONAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

// Original kernel — kept for RAJA_HIP variant (uses ZONAL_ACCUMULATION_3D macros)
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void zonal_accumulation_3d(Real_ptr vol,
                      Real_ptr x0, Real_ptr x1,
                      Real_ptr x2, Real_ptr x3,
                      Real_ptr x4, Real_ptr x5,
                      Real_ptr x6, Real_ptr x7,
                      Index_ptr real_zones,
                      Index_type ibegin, Index_type iend)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i = ii + ibegin;
   if (i < iend) {
     ZONAL_ACCUMULATION_3D_BODY_INDEX;
     ZONAL_ACCUMULATION_3D_BODY;
   }
}

// Leo optimization: Reduce kernel arguments and eliminate real_zones indirection.
//   1. Reduce kernel arguments: The original kernel passes 8 separate pointer
//      arguments (x0..x7), each requiring an s_load_dwordx2 to read from
//      scalar memory. Since x0..x7 are all offsets into the same base array x,
//      we pass only x + jp + kp (3 args instead of 10 pointer args). This
//      eliminates 7 s_load_dwordx2 instructions and the s_waitcnt lgkmcnt(0)
//      stalls that wait for them (23.1% + 7.7% = 30.8% of stall cycles).
//   2. Eliminate real_zones indirection: The real_zones[] array maps a compact
//      thread index to a zone index in the padded 3D mesh, but follows a
//      perfectly regular 3D pattern. We compute the zone index arithmetically
//      from the thread index, eliminating one global_load_dwordx2 and the
//      serialization it causes (all 8 stencil loads depend on real_zones[ii]).
//      This removes a critical dependency chain.
//   3. __restrict__ on all pointers: Enables the compiler to use read-only
//      cache paths and avoid aliasing-related reloads.
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void zonal_accumulation_3d_opt(
    Real_type* __restrict__ vol,
    Real_type* __restrict__ x,
    Index_type jp, Index_type kp,
    Index_type imin, Index_type jmin, Index_type kmin,
    Index_type ni, Index_type nj,
    Index_type iend)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   if (ii < iend) {
     // Compute zone index arithmetically (replaces real_zones[ii] load)
     Index_type li = ii % ni;
     Index_type lj = (ii / ni) % nj;
     Index_type lk = ii / (ni * nj);
     Index_type i = (li + imin) + (lj + jmin) * jp + (lk + kmin) * kp;

     // 8 corner values from single base pointer + stencil offsets
     vol[i] = 0.125 * ( x[i]              +
                        x[i + 1]          +
                        x[i + jp]         +
                        x[i + 1 + jp]     +
                        x[i + kp]         +
                        x[i + 1 + kp]     +
                        x[i + jp + kp]    +
                        x[i + 1 + jp + kp]
                      );
   }
}


template < size_t block_size >
void ZONAL_ACCUMULATION_3D::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  auto res{getHipResource()};

  ZONAL_ACCUMULATION_3D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    // Leo optimization: reduced args + no indirection + __restrict__
    const Index_type jp = m_domain->jp;
    const Index_type kp = m_domain->kp;
    const Index_type imin = m_domain->imin;
    const Index_type jmin = m_domain->jmin;
    const Index_type kmin = m_domain->kmin;
    const Index_type ni = m_domain->imax - m_domain->imin;
    const Index_type nj = m_domain->jmax - m_domain->jmin;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (zonal_accumulation_3d_opt<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         vol,
                         x,
                         jp, kp,
                         imin, jmin, kmin,
                         ni, nj,
                         iend );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                             res, RAJA::Unowned);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        zones, [=] __device__ (Index_type i) {
          ZONAL_ACCUMULATION_3D_BODY;
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  ZONAL_ACCUMULATION_3D : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(ZONAL_ACCUMULATION_3D, Hip, Base_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
