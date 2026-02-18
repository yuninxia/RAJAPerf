//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DEL_DOT_VEC_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include "AppsData.hpp"

#include "camp/resource.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void deldotvec2d(Real_ptr div,
                            const Real_ptr x1, const Real_ptr x2,
                            const Real_ptr x3, const Real_ptr x4,
                            const Real_ptr y1, const Real_ptr y2,
                            const Real_ptr y3, const Real_ptr y4,
                            const Real_ptr fx1, const Real_ptr fx2,
                            const Real_ptr fx3, const Real_ptr fx4,
                            const Real_ptr fy1, const Real_ptr fy2,
                            const Real_ptr fy3, const Real_ptr fy4,
                            const Index_ptr real_zones,
                            const Real_type half, const Real_type ptiny,
                            Index_type iend)
{
   Index_type ii = blockIdx.x * block_size + threadIdx.x;
   if (ii < iend) {
     DEL_DOT_VEC_2D_BODY_INDEX;
     DEL_DOT_VEC_2D_BODY;
   }
}

// Leo-optimized kernel: 4 base pointers + jp instead of 16 pre-offset pointers,
// __restrict__ for vectorized loads, occupancy hint, phased computation.
template < size_t block_size >
__launch_bounds__(block_size, 4)
__global__ void deldotvec2d_opt(
    Real_type* __restrict__ div,
    const Real_type* __restrict__ x,
    const Real_type* __restrict__ y,
    const Real_type* __restrict__ xdot,
    const Real_type* __restrict__ ydot,
    const Index_type* __restrict__ real_zones,
    Real_type half, Real_type ptiny,
    Index_type iend, Index_type jp)
{
   Index_type ii = blockIdx.x * block_size + threadIdx.x;
   if (ii < iend) {
     Index_type i = real_zones[ii];

     // Phase A: load coordinates, compute mesh gradients
     // NDSET2D: v4=v[i], v1=v[i+1], v2=v[i+1+jp], v3=v[i+jp]
     Real_type x4v = x[i],    x1v = x[i+1],    x2v = x[i+1+jp],  x3v = x[i+jp];
     Real_type y4v = y[i],    y1v = y[i+1],    y2v = y[i+1+jp],  y3v = y[i+jp];

     Real_type xi  = half * ( x1v + x2v - x3v - x4v ) ;
     Real_type xj  = half * ( x2v + x3v - x4v - x1v ) ;
     Real_type yi  = half * ( y1v + y2v - y3v - y4v ) ;
     Real_type yj  = half * ( y2v + y3v - y4v - y1v ) ;
     Real_type y_sum = y1v + y2v + y3v + y4v;

     // Phase B: load forces, compute divergence
     Real_type fx4v = xdot[i], fx1v = xdot[i+1], fx2v = xdot[i+1+jp], fx3v = xdot[i+jp];
     Real_type fy4v = ydot[i], fy1v = ydot[i+1], fy2v = ydot[i+1+jp], fy3v = ydot[i+jp];

     Real_type fxi = half * ( fx1v + fx2v - fx3v - fx4v ) ;
     Real_type fxj = half * ( fx2v + fx3v - fx4v - fx1v ) ;
     Real_type fyi = half * ( fy1v + fy2v - fy3v - fy4v ) ;
     Real_type fyj = half * ( fy2v + fy3v - fy4v - fy1v ) ;
     Real_type fy_sum = fy1v + fy2v + fy3v + fy4v;

     Real_type rarea  = 1.0 / ( xi * yj - xj * yi + ptiny ) ;
     Real_type dfxdx  = rarea * ( fxi * yj - fxj * yi ) ;
     Real_type dfydy  = rarea * ( fyj * xi - fyi * xj ) ;
     Real_type affine = fy_sum / y_sum ;

     div[i] = dfxdx + dfydy + affine ;
   }
}


template < size_t block_size >
void DEL_DOT_VEC_2D::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  auto res{getHipResource()};

  DEL_DOT_VEC_2D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const Index_type jp = m_domain->jp;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      const Real_type* cx = x;
      const Real_type* cy = y;
      const Real_type* cxdot = xdot;
      const Real_type* cydot = ydot;
      const Index_type* czones = real_zones;

      RPlaunchHipKernel( (deldotvec2d_opt<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         div,
                         cx, cy,
                         cxdot, cydot,
                         czones,
                         half, ptiny,
                         iend, jp );

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      auto deldotvec2d_lambda = [=] __device__ (Index_type ii) {
        DEL_DOT_VEC_2D_BODY_INDEX;
        DEL_DOT_VEC_2D_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (lambda_hip_forall<block_size,
                                            decltype(deldotvec2d_lambda)>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         ibegin, iend,
                         deldotvec2d_lambda );

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
         DEL_DOT_VEC_2D_BODY;
       });

    }
    stopTimer();

  } else {
     getCout() << "\n  DEL_DOT_VEC_2D : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DEL_DOT_VEC_2D, Hip, Base_HIP, Lambda_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
