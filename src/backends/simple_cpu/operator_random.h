#pragma once
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/opp_defs.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <random>

namespace opp
{
template <AnyVector T> struct FillRandom
{
    std::default_random_engine engine;

    FillRandom() = default;

    FillRandom(uint aSeed) : engine(aSeed)
    {
    }

    void operator()(T &aDst)
        requires NativeFloatingPoint<remove_vector_t<T>>
    {
        std::uniform_real_distribution<remove_vector_t<T>> uniform_dist(0, 1);
        aDst.x = uniform_dist(engine);
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = uniform_dist(engine);
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = uniform_dist(engine);
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = uniform_dist(engine);
        }
    }
    void operator()(T &aDst)
        requires IsHalfFp16<remove_vector_t<T>>
    {
        std::uniform_real_distribution<float> uniform_dist(0, 1);
        aDst.x = static_cast<HalfFp16>(uniform_dist(engine));
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = static_cast<HalfFp16>(uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = static_cast<HalfFp16>(uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = static_cast<HalfFp16>(uniform_dist(engine));
        }
    }
    void operator()(T &aDst)
        requires IsBFloat16<remove_vector_t<T>>
    {
        std::uniform_real_distribution<float> uniform_dist(0, 1);
        aDst.x = static_cast<BFloat16>(uniform_dist(engine));
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = static_cast<BFloat16>(uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = static_cast<BFloat16>(uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = static_cast<BFloat16>(uniform_dist(engine));
        }
    }
    void operator()(T &aDst)
        requires NativeIntegral<remove_vector_t<T>> && (!ByteSizeType<remove_vector_t<T>>)
    {
        std::uniform_int_distribution<remove_vector_t<T>> uniform_dist(numeric_limits<remove_vector_t<T>>::lowest(),
                                                                       numeric_limits<remove_vector_t<T>>::max());
        aDst.x = uniform_dist(engine);
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = uniform_dist(engine);
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = uniform_dist(engine);
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = uniform_dist(engine);
        }
    }
    void operator()(T &aDst)
        requires NativeIntegral<remove_vector_t<T>> && (ByteSizeType<remove_vector_t<T>>)
    {
        std::uniform_int_distribution<int> uniform_dist(numeric_limits<remove_vector_t<T>>::lowest(),
                                                        numeric_limits<remove_vector_t<T>>::max());
        aDst.x = static_cast<remove_vector_t<T>>(uniform_dist(engine));
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = static_cast<remove_vector_t<T>>(uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = static_cast<remove_vector_t<T>>(uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = static_cast<remove_vector_t<T>>(uniform_dist(engine));
        }
    }
    void operator()(T &aDst)
        requires ComplexFloatingPoint<remove_vector_t<T>>
    {
        std::uniform_real_distribution<complex_basetype_t<remove_vector_t<T>>> uniform_dist(0, 1);
        aDst.x = remove_vector_t<T>(uniform_dist(engine), uniform_dist(engine));
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = remove_vector_t<T>(uniform_dist(engine), uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = remove_vector_t<T>(uniform_dist(engine), uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = remove_vector_t<T>(uniform_dist(engine), uniform_dist(engine));
        }
    }
    void operator()(T &aDst)
        requires ComplexIntegral<remove_vector_t<T>>
    {
        std::uniform_int_distribution<complex_basetype_t<remove_vector_t<T>>> uniform_dist(
            numeric_limits<complex_basetype_t<remove_vector_t<T>>>::lowest(),
            numeric_limits<complex_basetype_t<remove_vector_t<T>>>::max());
        aDst.x = remove_vector_t<T>(uniform_dist(engine), uniform_dist(engine));
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = remove_vector_t<T>(uniform_dist(engine), uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = remove_vector_t<T>(uniform_dist(engine), uniform_dist(engine));
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = remove_vector_t<T>(uniform_dist(engine), uniform_dist(engine));
        }
    }
};
} // namespace opp