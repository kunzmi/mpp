#include "bfloat16.h"     //NOLINT(misc-include-cleaner)
#include "complex.h"      //NOLINT
#include "complex_impl.h" //NOLINT(misc-include-cleaner)
#include "defines.h"
#include "dllexport_common.h"
#include "half_fp16.h" //NOLINT(misc-include-cleaner)
#include "mpp_defs.h"  //NOLINT(misc-include-cleaner)

// Note: we instantiate all complex types we support in MPP here, so that we only need to include the normal header file
// in all host code

namespace mpp
{
template struct Complex<sbyte>;
template struct Complex<short>;
template struct Complex<int>;
template struct Complex<long64>;
template struct Complex<float>;
template struct Complex<double>;
template struct Complex<HalfFp16>;
template struct Complex<BFloat16>;

template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<float> &, RoundingMode) noexcept;

template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<float> &, RoundingMode) noexcept;

template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<short>::Complex(Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<short>::Complex(Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<int>::Complex(Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<int>::Complex(Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<float>::Complex(Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<float>::Complex(Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<double>::Complex(Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(Complex<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Complex<double>::Complex(Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<BFloat16> &) noexcept;

template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<sbyte> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<short> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<int> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<long64> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<float> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<double> &) noexcept;
template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<HalfFp16> &) noexcept;

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<mpp::sbyte> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<mpp::sbyte> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<mpp::sbyte> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<mpp::sbyte> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<short> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<short> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<short> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<short> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<long64> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<long64> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<long64> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<long64> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<int> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<int> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<int> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<int> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<float> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<float> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<float> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<float> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<double> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<double> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<double> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<double> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<mpp::HalfFp16> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<mpp::HalfFp16> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<mpp::HalfFp16> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<mpp::HalfFp16> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<mpp::BFloat16> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<mpp::BFloat16> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<mpp::BFloat16> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<mpp::BFloat16> &aVec);

} // namespace mpp