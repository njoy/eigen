// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TEST_ANNOYING_SCALAR_H
#define EIGEN_TEST_ANNOYING_SCALAR_H

#include <ostream>

#if EIGEN_COMP_GNUC
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#if defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW)
struct my_exception {
  my_exception() {}
  ~my_exception() {}
};
#endif

// An AnnoyingScalar is a pseudo scalar type that:
// - can randomly through an exception in operator +
// - randomly allocate on the heap or initialize a reference to itself making it non trivially copyable, nor movable,
// nor relocatable.

class AnnoyingScalar {
 public:
  AnnoyingScalar() {
    init();
    *v = 0;
  }
  AnnoyingScalar(long double _v) {
    init();
    *v = static_cast<float>(_v);
  }
  AnnoyingScalar(double _v) {
    init();
    *v = static_cast<float>(_v);
  }
  AnnoyingScalar(float _v) {
    init();
    *v = _v;
  }
  AnnoyingScalar(int _v) {
    init();
    *v = static_cast<float>(_v);
  }
  AnnoyingScalar(long _v) {
    init();
    *v = static_cast<float>(_v);
  }
  AnnoyingScalar(long long _v) {
    init();
    *v = static_cast<float>(_v);
  }
  AnnoyingScalar(const AnnoyingScalar& other) {
    init();
    *v = *(other.v);
  }
  ~AnnoyingScalar() {
    if (v != &data) delete v;
    instances--;
  }

  void init() {
    if (internal::random<bool>())
      v = new float;
    else
      v = &data;
    instances++;
  }

  AnnoyingScalar operator+(const AnnoyingScalar& other) const {
#if defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW)
    countdown--;
    if (countdown <= 0 && !dont_throw) throw my_exception();
#endif
    return AnnoyingScalar(*v + *other.v);
  }

  AnnoyingScalar operator-() const { return AnnoyingScalar(-*v); }

  AnnoyingScalar operator-(const AnnoyingScalar& other) const { return AnnoyingScalar(*v - *other.v); }

  AnnoyingScalar operator*(const AnnoyingScalar& other) const { return AnnoyingScalar((*v) * (*other.v)); }

  AnnoyingScalar operator/(const AnnoyingScalar& other) const { return AnnoyingScalar((*v) / (*other.v)); }

  AnnoyingScalar& operator+=(const AnnoyingScalar& other) {
    *v += *other.v;
    return *this;
  }
  AnnoyingScalar& operator-=(const AnnoyingScalar& other) {
    *v -= *other.v;
    return *this;
  }
  AnnoyingScalar& operator*=(const AnnoyingScalar& other) {
    *v *= *other.v;
    return *this;
  }
  AnnoyingScalar& operator/=(const AnnoyingScalar& other) {
    *v /= *other.v;
    return *this;
  }
  AnnoyingScalar& operator=(const AnnoyingScalar& other) {
    *v = *other.v;
    return *this;
  }

  bool operator==(const AnnoyingScalar& other) const { return numext::equal_strict(*v, *other.v); }
  bool operator!=(const AnnoyingScalar& other) const { return numext::not_equal_strict(*v, *other.v); }
  bool operator<=(const AnnoyingScalar& other) const { return *v <= *other.v; }
  bool operator<(const AnnoyingScalar& other) const { return *v < *other.v; }
  bool operator>=(const AnnoyingScalar& other) const { return *v >= *other.v; }
  bool operator>(const AnnoyingScalar& other) const { return *v > *other.v; }

  float* v;
  float data;
  static int instances;
#ifndef EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
  static int countdown;
  static bool dont_throw;
#endif
};

AnnoyingScalar real(const AnnoyingScalar& x) { return x; }
AnnoyingScalar imag(const AnnoyingScalar&) { return 0; }
AnnoyingScalar conj(const AnnoyingScalar& x) { return x; }
AnnoyingScalar sqrt(const AnnoyingScalar& x) { return std::sqrt(*x.v); }
AnnoyingScalar abs(const AnnoyingScalar& x) { return std::abs(*x.v); }
AnnoyingScalar cos(const AnnoyingScalar& x) { return std::cos(*x.v); }
AnnoyingScalar sin(const AnnoyingScalar& x) { return std::sin(*x.v); }
AnnoyingScalar acos(const AnnoyingScalar& x) { return std::acos(*x.v); }
AnnoyingScalar atan2(const AnnoyingScalar& y, const AnnoyingScalar& x) { return std::atan2(*y.v, *x.v); }

std::ostream& operator<<(std::ostream& stream, const AnnoyingScalar& x) {
  stream << (*(x.v));
  return stream;
}

int AnnoyingScalar::instances = 0;

#ifndef EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
int AnnoyingScalar::countdown = 0;
bool AnnoyingScalar::dont_throw = false;
#endif

namespace Eigen {
template <>
struct NumTraits<AnnoyingScalar> : NumTraits<float> {
  enum {
    RequireInitialization = 1,
  };
  typedef AnnoyingScalar Real;
  typedef AnnoyingScalar Nested;
  typedef AnnoyingScalar Literal;
  typedef AnnoyingScalar NonInteger;
};

template <>
inline AnnoyingScalar test_precision<AnnoyingScalar>() {
  return test_precision<float>();
}

namespace numext {
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isfinite)(const AnnoyingScalar& x) {
  return (numext::isfinite)(*x.v);
}
}  // namespace numext

namespace internal {
template <>
EIGEN_STRONG_INLINE double cast(const AnnoyingScalar& x) {
  return double(*x.v);
}
template <>
EIGEN_STRONG_INLINE float cast(const AnnoyingScalar& x) {
  return *x.v;
}

}  // namespace internal
}  // namespace Eigen

AnnoyingScalar get_test_precision(const AnnoyingScalar&) { return Eigen::test_precision<AnnoyingScalar>(); }

AnnoyingScalar test_relative_error(const AnnoyingScalar& a, const AnnoyingScalar& b) {
  return test_relative_error(*a.v, *b.v);
}

inline bool test_isApprox(const AnnoyingScalar& a, const AnnoyingScalar& b) {
  return internal::isApprox(*a.v, *b.v, test_precision<float>());
}

inline bool test_isMuchSmallerThan(const AnnoyingScalar& a, const AnnoyingScalar& b) {
  return test_isMuchSmallerThan(*a.v, *b.v);
}

#endif  // EIGEN_TEST_ANNOYING_SCALAR_H
