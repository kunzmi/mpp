#pragma once
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/roundFunctor.h>
#include <common/vector_typetraits.h>

namespace mpp::image
{
template <typename T> struct scalefactor_type
{
    using type = complex_basetype_t<pixel_basetype_t<T>>; // get the scalar base type of a pixel (complex_base_t does
                                                          // nothing for non-complex types)
};

template <typename T> using scalefactor_t = typename scalefactor_type<T>::type;

// base struct for all image processing functors
template <bool LoadBeforeOp> struct ImageFunctor
{
    // indicates if the functor operates inplace and that the kernel is supposed to load the destination image pixel
    // before the functor call.
    static constexpr bool DoLoadBeforeOp = LoadBeforeOp;
};

} // namespace mpp::image
