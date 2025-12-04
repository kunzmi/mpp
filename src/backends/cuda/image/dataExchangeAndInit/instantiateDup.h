#pragma once

namespace mpp::image::cuda
{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateDup_For(type)                                                                                       \
    template ImageView<Pixel##type##C3> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C3>(                             \
        ImageView<Pixel##type##C3> & aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;                              \
    template ImageView<Pixel##type##C4> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C4>(                             \
        ImageView<Pixel##type##C4> & aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;                              \
    template ImageView<Pixel##type##C4A> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C4A>(                           \
        ImageView<Pixel##type##C4A> & aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateDupNoAlpha_For(type)                                                                                \
    template ImageView<Pixel##type##C3> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C3>(                             \
        ImageView<Pixel##type##C3> & aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;                              \
    template ImageView<Pixel##type##C4> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C4>(                             \
        ImageView<Pixel##type##C4> & aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;

} // namespace mpp::image::cuda