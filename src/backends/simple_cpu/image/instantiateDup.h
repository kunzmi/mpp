#pragma once

namespace mpp::image::cpuSimple
{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateDup_For(type)                                                                                       \
    template MPPEXPORT_SIMPLECPU ImageView<Pixel##type##C3> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C3>(         \
        ImageView<Pixel##type##C3> & aDst) const;                                                                      \
    template MPPEXPORT_SIMPLECPU ImageView<Pixel##type##C4> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C4>(         \
        ImageView<Pixel##type##C4> & aDst) const;                                                                      \
    template MPPEXPORT_SIMPLECPU ImageView<Pixel##type##C4A> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C4A>(       \
        ImageView<Pixel##type##C4A> & aDst) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateDupNoAlpha_For(type)                                                                                \
    template MPPEXPORT_SIMPLECPU ImageView<Pixel##type##C3> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C3>(         \
        ImageView<Pixel##type##C3> & aDst) const;                                                                      \
    template MPPEXPORT_SIMPLECPU ImageView<Pixel##type##C4> &ImageView<Pixel##type##C1>::Dup<Pixel##type##C4>(         \
        ImageView<Pixel##type##C4> & aDst) const;

} // namespace mpp::image::cpuSimple