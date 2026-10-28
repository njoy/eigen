// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define SCALAR double
#define SCALAR_SUFFIX d
#define SCALAR_SUFFIX_UP "D"
#define ISCOMPLEX 0

#include "level1_impl.h"
#include "level1_real_impl.h"
#include "level2_impl.h"
#include "level2_real_impl.h"
#include "level3_impl.h"

extern "C" double EIGEN_BLAS_FUNC_NAME(sdot)(EIGEN_BLAS_INT* n, float* x, EIGEN_BLAS_INT* incx, float* y,
                                             EIGEN_BLAS_INT* incy) {
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  const Eigen::Index incy_idx = *incy;

  if (n_idx <= 0) return 0;

  if (incx_idx == 1 && incy_idx == 1)
    return (make_vector(x, n_idx).cast<double>().cwiseProduct(make_vector(y, n_idx).cast<double>())).sum();
  else if (incx_idx >= 0 && incy_idx >= 0)
    return (make_vector(x, n_idx, incx_idx).cast<double>().cwiseProduct(make_vector(y, n_idx, incy_idx).cast<double>()))
        .sum();
  else if (incx_idx < 0 && incy_idx >= 0)
    return (make_vector(x, n_idx, -incx_idx)
                .reverse()
                .cast<double>()
                .cwiseProduct(make_vector(y, n_idx, incy_idx).cast<double>()))
        .sum();
  else if (incx_idx >= 0 && incy_idx < 0)
    return (make_vector(x, n_idx, incx_idx)
                .cast<double>()
                .cwiseProduct(make_vector(y, n_idx, -incy_idx).reverse().cast<double>()))
        .sum();
  else if (incx_idx < 0 && incy_idx < 0)
    return (make_vector(x, n_idx, -incx_idx)
                .reverse()
                .cast<double>()
                .cwiseProduct(make_vector(y, n_idx, -incy_idx).reverse().cast<double>()))
        .sum();
  else
    return 0;
}
