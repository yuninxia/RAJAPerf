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

#if defined(RAJA_ENABLE_SYCL)

#include "AppsData.hpp"

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

template <size_t work_group_size >
void DEL_DOT_VEC_2D::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  DEL_DOT_VEC_2D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    // Leo optimization: inline macro body with 4 base pointers + jp offset
    // instead of 16 pre-offset pointers, phased load/compute, and __restrict__.
    //
    // Root cause (Intel PVC, 72.7% stall ratio, est. 1.22x speedup):
    //   - urb self-stall (25%): URB write contention from 16 aliased pointer streams
    //   - wait stalled by math (25%): memory fences interleave with FP ops
    //   - madm self-stall (12.5%): register pressure from redundant expressions
    //   - math stalled by wait (12.5%): FP unit starved behind fence latency
    //   - smov stalled by wait (12.5%): scalar moves blocked by unnecessary fences
    //
    // Fix: reduce to 4 base pointers with __restrict__ (eliminates aliasing fences),
    // phase coordinate loads before force loads (better URB scheduling), and
    // precompute shared sub-expressions (y_sum, fy_sum) to cut register pressure.
    const Index_type jp = m_domain->jp;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type ii = item.get_global_id(0);
          if (ii < iend) {
            Index_type i = real_zones[ii];

            // Phase A: load coordinates, compute mesh gradients
            // NDSET2D layout: v4=v[i], v1=v[i+1], v2=v[i+1+jp], v3=v[i+jp]
            Real_type x4v = x[i],    x1v = x[i+1],    x2v = x[i+1+jp],  x3v = x[i+jp];
            Real_type y4v = y[i],    y1v = y[i+1],    y2v = y[i+1+jp],  y3v = y[i+jp];

            Real_type xi  = half * ( x1v + x2v - x3v - x4v );
            Real_type xj  = half * ( x2v + x3v - x4v - x1v );
            Real_type yi  = half * ( y1v + y2v - y3v - y4v );
            Real_type yj  = half * ( y2v + y3v - y4v - y1v );
            Real_type y_sum = y1v + y2v + y3v + y4v;

            // Phase B: load forces, compute divergence
            Real_type fx4v = xdot[i], fx1v = xdot[i+1], fx2v = xdot[i+1+jp], fx3v = xdot[i+jp];
            Real_type fy4v = ydot[i], fy1v = ydot[i+1], fy2v = ydot[i+1+jp], fy3v = ydot[i+jp];

            Real_type fxi = half * ( fx1v + fx2v - fx3v - fx4v );
            Real_type fxj = half * ( fx2v + fx3v - fx4v - fx1v );
            Real_type fyi = half * ( fy1v + fy2v - fy3v - fy4v );
            Real_type fyj = half * ( fy2v + fy3v - fy4v - fy1v );
            Real_type fy_sum = fy1v + fy2v + fy3v + fy4v;

            Real_type rarea  = 1.0 / ( xi * yj - xj * yi + ptiny );
            Real_type dfxdx  = rarea * ( fxi * yj - fxj * yi );
            Real_type dfydy  = rarea * ( fyj * xi - fyi * xj );
            Real_type affine = fy_sum / y_sum;

            div[i] = dfxdx + dfydy + affine;
          }

        });
      });

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                             res, RAJA::Unowned);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >( res,
         zones, [=] (Index_type i) {
         DEL_DOT_VEC_2D_BODY;
       });

    }
    stopTimer();

  } else {
     std::cout << "\n  DEL_DOT_VEC_2D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DEL_DOT_VEC_2D, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
