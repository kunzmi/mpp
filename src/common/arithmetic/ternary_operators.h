#pragma once
#include <common/defines.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp
{
template <VectorType T> struct AddMul
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aSrcDst)
    {
        aSrcDst += aSrc1 * aSrc2;
    }
};

template <VectorType T> struct AddWeighted
{
    remove_vector_t<T> Alpha;

    AddWeighted(remove_vector_t<T> aAlpha) : Alpha(aAlpha)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aResult)
    {
        aResult = aSrc1 * Alpha + aSrc2 * (1.0f - Alpha);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst = aSrcDst * (1.0f - Alpha) + aSrc1 * Alpha;
    }
};
} // namespace opp
