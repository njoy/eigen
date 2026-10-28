// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "common.h"

EIGEN_BLAS_FUNC(axpy)
(const EIGEN_BLAS_INT *n, const RealScalar *palpha, const RealScalar *px, const EIGEN_BLAS_INT *incx, RealScalar *py,
 const EIGEN_BLAS_INT *incy) {
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);

  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  const Eigen::Index incy_idx = *incy;

  if (n_idx <= 0) return;

  if (incx_idx == 1 && incy_idx == 1)
    make_vector(y, n_idx) += alpha * make_vector(x, n_idx);
  else if (incx_idx >= 0 && incy_idx >= 0)
    make_vector(y, n_idx, incy_idx) += alpha * make_vector(x, n_idx, incx_idx);
  else if (incx_idx >= 0 && incy_idx < 0)
    make_vector(y, n_idx, -incy_idx) += alpha * make_vector(x, n_idx, incx_idx).reverse();
  else if (incx_idx < 0 && incy_idx >= 0)
    make_vector(y, n_idx, incy_idx) += alpha * make_vector(x, n_idx, -incx_idx).reverse();
  else if (incx_idx < 0 && incy_idx < 0)
    make_vector(y, n_idx, -incy_idx) += alpha * make_vector(x, n_idx, -incx_idx);
}

EIGEN_BLAS_FUNC(axpby)
(const EIGEN_BLAS_INT *pn, const RealScalar *palpha, const RealScalar *px, const EIGEN_BLAS_INT *pincx,
 const RealScalar *pbeta, RealScalar *py, const EIGEN_BLAS_INT *pincy) {
  const Scalar *x = reinterpret_cast<const Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  const Scalar alpha = *reinterpret_cast<const Scalar *>(palpha);
  const Scalar beta = *reinterpret_cast<const Scalar *>(pbeta);
  const Eigen::Index n = *pn;

  if (n <= 0) return;

  if (Eigen::numext::equal_strict(beta, Scalar(1))) {
    EIGEN_BLAS_FUNC_NAME(axpy)(pn, palpha, px, pincx, py, pincy);
    return;
  }

  const Eigen::Index incx = *pincx;
  const Eigen::Index incy = *pincy;

  if (incx == 1 && incy == 1)
    make_vector(y, n) = alpha * make_vector(x, n) + beta * make_vector(y, n);
  else if (incx > 0 && incy > 0)
    make_vector(y, n, incy) = alpha * make_vector(x, n, incx) + beta * make_vector(y, n, incy);
  else if (incx > 0 && incy < 0)
    make_vector(y, n, -incy) = alpha * make_vector(x, n, incx).reverse() + beta * make_vector(y, n, -incy);
  else if (incx < 0 && incy > 0)
    make_vector(y, n, incy) = alpha * make_vector(x, n, -incx).reverse() + beta * make_vector(y, n, incy);
  else if (incx < 0 && incy < 0)
    make_vector(y, n, -incy) = alpha * make_vector(x, n, -incx) + beta * make_vector(y, n, -incy);
}

EIGEN_BLAS_FUNC(copy)(EIGEN_BLAS_INT *n, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py, EIGEN_BLAS_INT *incy) {
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  const Eigen::Index incy_idx = *incy;

  if (n_idx <= 0) return;

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);

  // be careful, *incx==0 is allowed !!
  if (incx_idx == 1 && incy_idx == 1)
    make_vector(y, n_idx) = make_vector(x, n_idx);
  else if (incx_idx == 0) {
    // Broadcast: copy x[0] to all elements of y.
    if (incy_idx < 0) y = y - (n_idx - 1) * incy_idx;
    for (Eigen::Index i = 0; i < n_idx; ++i) {
      *y = *x;
      y += incy_idx;
    }
  } else if (incx_idx > 0 && incy_idx > 0)
    make_vector(y, n_idx, incy_idx) = make_vector(x, n_idx, incx_idx);
  else if (incx_idx > 0 && incy_idx < 0)
    make_vector(y, n_idx, -incy_idx).reverse() = make_vector(x, n_idx, incx_idx);
  else if (incx_idx < 0 && incy_idx > 0)
    make_vector(y, n_idx, incy_idx) = make_vector(x, n_idx, -incx_idx).reverse();
  else if (incx_idx < 0 && incy_idx < 0)
    make_vector(y, n_idx, -incy_idx) = make_vector(x, n_idx, -incx_idx);
}

EIGEN_BLAS_FUNC(rotg)(RealScalar *pa, RealScalar *pb, RealScalar *pc, RealScalar *ps) {
  using std::abs;
  using std::sqrt;

  Scalar &a = *reinterpret_cast<Scalar *>(pa);
  Scalar &b = *reinterpret_cast<Scalar *>(pb);
  RealScalar *c = pc;
  Scalar *s = reinterpret_cast<Scalar *>(ps);

#if !ISCOMPLEX
  Scalar r, z;
  Scalar aa = abs(a);
  Scalar ab = abs(b);
  if ((aa + ab) == Scalar(0)) {
    *c = 1;
    *s = 0;
    r = 0;
    z = 0;
  } else {
    r = Eigen::numext::hypot(a, b);
    Scalar amax = aa > ab ? a : b;
    r = amax > 0 ? r : -r;
    *c = a / r;
    *s = b / r;
    z = 1;
    if (aa > ab) z = *s;
    if (ab >= aa && *c != RealScalar(0)) z = Scalar(1) / *c;
  }
  *pa = r;
  *pb = z;
#else
  Scalar alpha;
  RealScalar norm, scale;
  if (abs(a) == RealScalar(0)) {
    *c = RealScalar(0);
    *s = Scalar(1);
    a = b;
  } else {
    scale = abs(a) + abs(b);
    norm = scale * sqrt((Eigen::numext::abs2(a / scale)) + (Eigen::numext::abs2(b / scale)));
    alpha = a / abs(a);
    *c = abs(a) / norm;
    *s = alpha * Eigen::numext::conj(b) / norm;
    a = alpha * norm;
  }
#endif

  //   JacobiRotation<Scalar> r;
  //   r.makeGivens(a,b);
  //   *c = r.c();
  //   *s = r.s();
}

EIGEN_BLAS_FUNC(scal)(EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx) {
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  if (n_idx <= 0 || incx_idx <= 0) return;

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar alpha = *reinterpret_cast<Scalar *>(palpha);

  if (incx_idx == 1)
    make_vector(x, n_idx) *= alpha;
  else
    make_vector(x, n_idx, std::abs(incx_idx)) *= alpha;
}

EIGEN_BLAS_FUNC(swap)(EIGEN_BLAS_INT *n, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py, EIGEN_BLAS_INT *incy) {
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  const Eigen::Index incy_idx = *incy;
  if (n_idx <= 0) return;

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);

  if (incx_idx == 1 && incy_idx == 1)
    make_vector(y, n_idx).swap(make_vector(x, n_idx));
  else if (incx_idx > 0 && incy_idx > 0)
    make_vector(y, n_idx, incy_idx).swap(make_vector(x, n_idx, incx_idx));
  else if (incx_idx > 0 && incy_idx < 0)
    make_vector(y, n_idx, -incy_idx).reverse().swap(make_vector(x, n_idx, incx_idx));
  else if (incx_idx < 0 && incy_idx > 0)
    make_vector(y, n_idx, incy_idx).swap(make_vector(x, n_idx, -incx_idx).reverse());
  else if (incx_idx < 0 && incy_idx < 0)
    make_vector(y, n_idx, -incy_idx).reverse().swap(make_vector(x, n_idx, -incx_idx).reverse());
}
