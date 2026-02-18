//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "VOL3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

// Leo optimization: base pointers + __restrict__ + launch_bounds(,4) + phased loads.
// Original: 24 pre-offset pointers passed via NDPTRSET (high SGPR/VGPR pressure,
// memory-latency stalls on AMD MI300A -- 73.7% stalls from global_load_dwordx2).
// Optimized: 3 base pointers + 2 strides (27->8 kernel args, SGPR 74->~30),
// phased loads (x then y then z) so corner registers die between phases,
// single local accumulation variable, single global store at end.
template < size_t block_size >
__launch_bounds__(block_size, 4)
__global__ void vol3d(Real_type* __restrict__ vol,
                      const Real_type* __restrict__ x,
                      const Real_type* __restrict__ y,
                      const Real_type* __restrict__ z,
                      const Real_type vnormq,
                      const Index_type jp, const Index_type kp,
                      Index_type ibegin, Index_type iend)
{
   Index_type ii = blockIdx.x * block_size + threadIdx.x;
   Index_type i = ii + ibegin;
   if (i < iend) {
     // Phase A -- load x corners via base ptr + offsets, compute x diffs
     Real_type x0v = x[i];
     Real_type x1v = x[i + 1];
     Real_type x2v = x[i + jp];
     Real_type x3v = x[i + 1 + jp];
     Real_type x4v = x[i + kp];
     Real_type x5v = x[i + 1 + kp];
     Real_type x6v = x[i + jp + kp];
     Real_type x7v = x[i + 1 + jp + kp];

     Real_type x71 = x7v - x1v;
     Real_type x72 = x7v - x2v;
     Real_type x74 = x7v - x4v;
     Real_type x30 = x3v - x0v;
     Real_type x50 = x5v - x0v;
     Real_type x60 = x6v - x0v;
     // x0v..x7v dead here -- compiler can reuse registers

     // Phase B -- load y corners, compute y diffs
     Real_type y0v = y[i];
     Real_type y1v = y[i + 1];
     Real_type y2v = y[i + jp];
     Real_type y3v = y[i + 1 + jp];
     Real_type y4v = y[i + kp];
     Real_type y5v = y[i + 1 + kp];
     Real_type y6v = y[i + jp + kp];
     Real_type y7v = y[i + 1 + jp + kp];

     Real_type y71 = y7v - y1v;
     Real_type y72 = y7v - y2v;
     Real_type y74 = y7v - y4v;
     Real_type y30 = y3v - y0v;
     Real_type y50 = y5v - y0v;
     Real_type y60 = y6v - y0v;
     // y0v..y7v dead here

     // Phase C -- load z corners, compute z diffs
     Real_type z0v = z[i];
     Real_type z1v = z[i + 1];
     Real_type z2v = z[i + jp];
     Real_type z3v = z[i + 1 + jp];
     Real_type z4v = z[i + kp];
     Real_type z5v = z[i + 1 + kp];
     Real_type z6v = z[i + jp + kp];
     Real_type z7v = z[i + 1 + jp + kp];

     Real_type z71 = z7v - z1v;
     Real_type z72 = z7v - z2v;
     Real_type z74 = z7v - z4v;
     Real_type z30 = z3v - z0v;
     Real_type z50 = z5v - z0v;
     Real_type z60 = z6v - z0v;
     // z0v..z7v dead here

     // Cross products -- accumulate in local variable, single store at end
     Real_type xps = x71 + x60;
     Real_type yps = y71 + y60;
     Real_type zps = z71 + z60;

     Real_type cyz = y72 * z30 - z72 * y30;
     Real_type czx = z72 * x30 - x72 * z30;
     Real_type cxy = x72 * y30 - y72 * x30;
     Real_type v = xps * cyz + yps * czx + zps * cxy;

     xps = x72 + x50;
     yps = y72 + y50;
     zps = z72 + z50;

     cyz = y74 * z60 - z74 * y60;
     czx = z74 * x60 - x74 * z60;
     cxy = x74 * y60 - y74 * x60;
     v += xps * cyz + yps * czx + zps * cxy;

     xps = x74 + x30;
     yps = y74 + y30;
     zps = z74 + z30;

     cyz = y71 * z50 - z71 * y50;
     czx = z71 * x50 - x71 * z50;
     cxy = x71 * y50 - y71 * x50;
     v += xps * cyz + yps * czx + zps * cxy;

     vol[i] = v * vnormq;
   }
}


template < size_t block_size >
void VOL3D::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  auto res{getHipResource()};

  VOL3D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const Index_type jp = m_domain->jp;
    const Index_type kp = m_domain->kp;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      const Real_type* cx = x;
      const Real_type* cy = y;
      const Real_type* cz = z;

      RPlaunchHipKernel( (vol3d<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         vol,
                         cx, cy, cz,
                         vnormq,
                         jp, kp,
                         ibegin, iend );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        VOL3D_BODY;
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  VOL3D : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(VOL3D, Hip, Base_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
