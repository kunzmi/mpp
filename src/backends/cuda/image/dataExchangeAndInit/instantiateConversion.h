#pragma once

namespace mpp::image::cuda
{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateConvert_For(typeSrc, typeDst)                                                                       \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Convert<typeDst>(                                 \
        ImageView<typeDst> & aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateConvertRound_For(typeSrc, typeDst)                                                                  \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Convert<typeDst>(                                 \
        ImageView<typeDst> & aDst, RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateConvertRoundScale_For(typeSrc, typeDst)                                                             \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Convert<typeDst>(                                 \
        ImageView<typeDst> & aDst, RoundingMode aRoundingMode, int aScaleFactor,                                       \
        const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsConvertWithAlpha(typeSrc, typeDst)                                                               \
    InstantiateConvert_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                    \
    InstantiateConvert_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                    \
    InstantiateConvert_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                    \
    InstantiateConvert_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                                    \
    InstantiateConvert_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsConvertNoAlpha(typeSrc, typeDst)                                                                 \
    InstantiateConvert_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                    \
    InstantiateConvert_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                    \
    InstantiateConvert_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                    \
    InstantiateConvert_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsConvertRoundWithAlpha(typeSrc, typeDst)                                                          \
    InstantiateConvertRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                               \
    InstantiateConvertRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                               \
    InstantiateConvertRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                               \
    InstantiateConvertRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                               \
    InstantiateConvertRound_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsConvertRoundNoAlpha(typeSrc, typeDst)                                                            \
    InstantiateConvertRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                               \
    InstantiateConvertRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                               \
    InstantiateConvertRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                               \
    InstantiateConvertRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsConvertRoundScaleWithAlpha(typeSrc, typeDst)                                                     \
    InstantiateConvertRoundScale_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                          \
    InstantiateConvertRoundScale_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                          \
    InstantiateConvertRoundScale_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                          \
    InstantiateConvertRoundScale_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                          \
    InstantiateConvertRoundScale_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsConvertRoundScaleNoAlpha(typeSrc, typeDst)                                                       \
    InstantiateConvertRoundScale_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                          \
    InstantiateConvertRoundScale_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                          \
    InstantiateConvertRoundScale_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                          \
    InstantiateConvertRoundScale_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

} // namespace mpp::image::cuda