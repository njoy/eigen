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

struct scalar_norm1_op {
  typedef RealScalar result_type;
  inline RealScalar operator()(const Scalar &a) const { return Eigen::numext::norm1(a); }
};
namespace Eigen {
namespace internal {
template <>
struct functor_traits<scalar_norm1_op> {
  enum { Cost = 3 * NumTraits<Scalar>::AddCost, PacketAccess = 0 };
};
}  // namespace internal
}  // namespace Eigen

// computes |Re(z)| + |Im(z)| of a complex number z.
extern "C" EIGEN_BLAS_API RealScalar EIGEN_CAT(REAL_SCALAR_SUFFIX,
                                               EIGEN_CAT(cabs1, EIGEN_BLAS_FUNC_SUFFIX))(Complex *z) {
  return Eigen::numext::norm1(*z);
}

// computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
// res = |Rex1| + |Imx1| + |Rex2| + |Imx2| + ... + |Rexn| + |Imxn|, where x is a vector of order n
extern "C" RealScalar EIGEN_CAT(REAL_SCALAR_SUFFIX, EIGEN_BLAS_FUNC_NAME(asum))(EIGEN_BLAS_INT *n, RealScalar *px,
                                                                                EIGEN_BLAS_INT *incx) {
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  if (n_idx <= 0 || incx_idx <= 0) return 0;

  // std::complex<T> is layout-compatible with T[2], so we can reinterpret
  // a complex vector of length n as a real vector of length 2*n and use
  // the fully vectorized cwiseAbs().sum() path.
  if (incx_idx == 1)
    return make_vector(px, 2 * n_idx).cwiseAbs().sum();
  else {
    // For non-unit stride, fall back to the scalar_norm1_op approach since
    // the real components are not contiguous across complex elements.
    Complex *x = reinterpret_cast<Complex *>(px);
    return make_vector(x, n_idx, std::abs(incx_idx)).unaryExpr<scalar_norm1_op>().sum();
  }
}

extern "C" EIGEN_BLAS_INT EIGEN_CAT(i, EIGEN_BLAS_FUNC_NAME(amax))(EIGEN_BLAS_INT *n, RealScalar *px,
                                                                   EIGEN_BLAS_INT *incx) {
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  if (n_idx <= 0 || incx_idx <= 0) return 0;
  Scalar *x = reinterpret_cast<Scalar *>(px);

  Eigen::DenseIndex ret;
  if (incx_idx == 1)
    make_vector(x, n_idx).unaryExpr<scalar_norm1_op>().maxCoeff(&ret);
  else
    make_vector(x, n_idx, std::abs(incx_idx)).unaryExpr<scalar_norm1_op>().maxCoeff(&ret);
  return EIGEN_BLAS_INT(ret) + 1;
}

extern "C" EIGEN_BLAS_INT EIGEN_CAT(i, EIGEN_BLAS_FUNC_NAME(amin))(EIGEN_BLAS_INT *n, RealScalar *px,
                                                                   EIGEN_BLAS_INT *incx) {
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  if (n_idx <= 0) return 0;
  Scalar *x = reinterpret_cast<Scalar *>(px);

  Eigen::DenseIndex ret;
  if (incx_idx == 1)
    make_vector(x, n_idx).unaryExpr<scalar_norm1_op>().minCoeff(&ret);
  else
    make_vector(x, n_idx, std::abs(incx_idx)).unaryExpr<scalar_norm1_op>().minCoeff(&ret);
  return EIGEN_BLAS_INT(ret) + 1;
}

// computes a dot product of a conjugated vector with another vector.
EIGEN_BLAS_FUNC(dotcw)
(EIGEN_BLAS_INT *n, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py, EIGEN_BLAS_INT *incy, RealScalar *pres) {
  //   std::cerr << "_dotc " << *n << " " << *incx << " " << *incy << "\n";
  Scalar *res = reinterpret_cast<Scalar *>(pres);

  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  const Eigen::Index incy_idx = *incy;

  if (n_idx <= 0) {
    *res = Scalar(0);
    return;
  }

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);

  if (incx_idx == 1 && incy_idx == 1)
    *res = (make_vector(x, n_idx).dot(make_vector(y, n_idx)));
  else if (incx_idx >= 0 && incy_idx >= 0)
    *res = (make_vector(x, n_idx, incx_idx).dot(make_vector(y, n_idx, incy_idx)));
  else if (incx_idx < 0 && incy_idx >= 0)
    *res = (make_vector(x, n_idx, -incx_idx).reverse().dot(make_vector(y, n_idx, incy_idx)));
  else if (incx_idx >= 0 && incy_idx < 0)
    *res = (make_vector(x, n_idx, incx_idx).dot(make_vector(y, n_idx, -incy_idx).reverse()));
  else if (incx_idx < 0 && incy_idx < 0)
    *res = (make_vector(x, n_idx, -incx_idx).reverse().dot(make_vector(y, n_idx, -incy_idx).reverse()));
}

// computes a vector-vector dot product without complex conjugation.
EIGEN_BLAS_FUNC(dotuw)
(EIGEN_BLAS_INT *n, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py, EIGEN_BLAS_INT *incy, RealScalar *pres) {
  Scalar *res = reinterpret_cast<Scalar *>(pres);

  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  const Eigen::Index incy_idx = *incy;

  if (n_idx <= 0) {
    *res = Scalar(0);
    return;
  }

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);

  if (incx_idx == 1 && incy_idx == 1)
    *res = (make_vector(x, n_idx).cwiseProduct(make_vector(y, n_idx))).sum();
  else if (incx_idx >= 0 && incy_idx >= 0)
    *res = (make_vector(x, n_idx, incx_idx).cwiseProduct(make_vector(y, n_idx, incy_idx))).sum();
  else if (incx_idx < 0 && incy_idx >= 0)
    *res = (make_vector(x, n_idx, -incx_idx).reverse().cwiseProduct(make_vector(y, n_idx, incy_idx))).sum();
  else if (incx_idx >= 0 && incy_idx < 0)
    *res = (make_vector(x, n_idx, incx_idx).cwiseProduct(make_vector(y, n_idx, -incy_idx).reverse())).sum();
  else if (incx_idx < 0 && incy_idx < 0)
    *res = (make_vector(x, n_idx, -incx_idx).reverse().cwiseProduct(make_vector(y, n_idx, -incy_idx).reverse())).sum();
}

extern "C" RealScalar EIGEN_CAT(REAL_SCALAR_SUFFIX, EIGEN_BLAS_FUNC_NAME(nrm2))(EIGEN_BLAS_INT *n, RealScalar *px,
                                                                                EIGEN_BLAS_INT *incx) {
  //   std::cerr << "__nrm2 " << *n << " " << *incx << "\n";
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  if (n_idx <= 0) return 0;

  Scalar *x = reinterpret_cast<Scalar *>(px);

  if (incx_idx == 1) return make_vector(x, n_idx).stableNorm();

  return make_vector(x, n_idx, std::abs(incx_idx)).stableNorm();
}

EIGEN_BLAS_FUNC(EIGEN_CAT(REAL_SCALAR_SUFFIX, rot))
(EIGEN_BLAS_INT *n, RealScalar *px, EIGEN_BLAS_INT *incx, RealScalar *py, EIGEN_BLAS_INT *incy, RealScalar *pc,
 RealScalar *ps) {
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  const Eigen::Index incy_idx = *incy;
  if (n_idx <= 0) return;

  Scalar *x = reinterpret_cast<Scalar *>(px);
  Scalar *y = reinterpret_cast<Scalar *>(py);
  RealScalar c = *pc;
  RealScalar s = *ps;

  StridedVectorType vx(make_vector(x, n_idx, std::abs(incx_idx)));
  StridedVectorType vy(make_vector(y, n_idx, std::abs(incy_idx)));

  Eigen::Reverse<StridedVectorType> rvx(vx);
  Eigen::Reverse<StridedVectorType> rvy(vy);

  // TODO implement mixed real-scalar rotations
  if (incx_idx < 0 && incy_idx > 0)
    Eigen::internal::apply_rotation_in_the_plane(rvx, vy, Eigen::JacobiRotation<Scalar>(c, s));
  else if (incx_idx > 0 && incy_idx < 0)
    Eigen::internal::apply_rotation_in_the_plane(vx, rvy, Eigen::JacobiRotation<Scalar>(c, s));
  else
    Eigen::internal::apply_rotation_in_the_plane(vx, vy, Eigen::JacobiRotation<Scalar>(c, s));
}

EIGEN_BLAS_FUNC(EIGEN_CAT(REAL_SCALAR_SUFFIX, scal))
(EIGEN_BLAS_INT *n, RealScalar *palpha, RealScalar *px, EIGEN_BLAS_INT *incx) {
  const Eigen::Index n_idx = *n;
  const Eigen::Index incx_idx = *incx;
  if (n_idx <= 0 || incx_idx <= 0) return;

  Scalar *x = reinterpret_cast<Scalar *>(px);
  RealScalar alpha = *palpha;

  //   std::cerr << "__scal " << *n << " " << alpha << " " << *incx << "\n";

  if (incx_idx == 1)
    make_vector(x, n_idx) *= alpha;
  else
    make_vector(x, n_idx, std::abs(incx_idx)) *= alpha;
}
