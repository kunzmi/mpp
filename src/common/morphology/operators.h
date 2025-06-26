#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/numeric_limits.h>
#include <common/mpp_defs.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace mpp
{
template <typename DstT, typename FilterT> struct Dilate
{
    static constexpr remove_vector_t<DstT> InitValue = numeric_limits<remove_vector_t<DstT>>::min();

    DEVICE_CODE void operator()(const FilterT &aFilter, const DstT &aSrc, DstT &aDst) const
    {
        if (aFilter > 0)
        {
            aDst.Max(aSrc);
        }
    }
};
template <typename DstT, typename FilterT> struct Erode
{
    static constexpr remove_vector_t<DstT> InitValue = numeric_limits<remove_vector_t<DstT>>::max();

    DEVICE_CODE void operator()(const FilterT &aFilter, const DstT &aSrc, DstT &aDst) const
    {
        if (aFilter > 0)
        {
            aDst.Min(aSrc);
        }
    }
};

template <typename DstT, typename FilterT> struct DilateGray
{
    static constexpr remove_vector_t<DstT> InitValue = numeric_limits<remove_vector_t<DstT>>::min();

    DEVICE_CODE void operator()(const FilterT &aFilter, const DstT &aSrc, DstT &aDst) const
    {
        // if (aFilter != static_cast<remove_vector_t<FilterT>>(0))
        {
            same_vector_size_different_type_t<DstT, remove_vector_t<FilterT>> srcPlusMask(aSrc);
            srcPlusMask += aFilter.x;
            DstT clampedValue(srcPlusMask);
            aDst.Max(clampedValue);
        }
    }
};
template <typename DstT, typename FilterT> struct ErodeGray
{
    static constexpr remove_vector_t<DstT> InitValue = numeric_limits<remove_vector_t<DstT>>::max();

    DEVICE_CODE void operator()(const FilterT &aFilter, const DstT &aSrc, DstT &aDst) const
    {
        // if (aFilter != static_cast<remove_vector_t<FilterT>>(0))
        {
            same_vector_size_different_type_t<DstT, remove_vector_t<FilterT>> srcPlusMask(aSrc);
            srcPlusMask += aFilter.x;
            DstT clampedValue(srcPlusMask);
            aDst.Min(clampedValue);
        }
    }
};

} // namespace mpp
