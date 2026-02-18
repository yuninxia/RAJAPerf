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

#if defined(RAJA_ENABLE_SYCL)

#include "AppsData.hpp"

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

template <size_t work_group_size >
void VOL3D::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  VOL3D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    // Leo optimization: base pointers + phased loads + local accumulation.
    // Original: 24 pre-offset pointers captured via NDPTRSET (register pressure,
    // sel stalls on Intel PVC -- 94.7% stall ratio, 1.46x estimated speedup).
    // Optimized: 3 base pointers + 2 strides, phased loads (x then y then z)
    // so corner registers die between phases, single store at end.
    const Index_type jp = m_domain->jp;
    const Index_type kp = m_domain->kp;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type ii = item.get_global_id(0);
          Index_type i = ii + ibegin;
          if (i < iend) {
            // OPT 1+2: Phase A -- load x corners via base ptr + offsets, compute x diffs
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

            // OPT 1+2: Phase B -- load y corners, compute y diffs
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

            // OPT 1+2: Phase C -- load z corners, compute z diffs
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

            // OPT 3: Cross products -- accumulate in local variable
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

            // Single global store
            vol[i] = v * vnormq;
          }

        });
      });

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        VOL3D_BODY;
      });

    }
    stopTimer();

  } else {
     std::cout << "\n  VOL3D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(VOL3D, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
