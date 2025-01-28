#pragma once
#include <common/defines.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp
{
template <AnyVector T> struct AddProduct
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aSrcDst)
    {
        aSrcDst += aSrc1 * aSrc2;
    }
};

template <AnyVector T> struct AddWeighted
{
    const remove_vector_t<T> Alpha;

    AddWeighted(remove_vector_t<T> aAlpha) : Alpha(aAlpha)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aResult)
    {
        aResult = aSrc1 * Alpha + aSrc2 * (static_cast<remove_vector_t<T>>(1) - Alpha);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst = aSrcDst * (static_cast<remove_vector_t<T>>(1) - Alpha) + aSrc1 * Alpha;
    }
};
} // namespace opp
