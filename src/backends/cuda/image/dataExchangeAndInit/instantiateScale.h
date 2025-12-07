#pragma once

namespace mpp::image::cuda
{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateScaleIntToInt_For(typeSrc, typeDst)                                                                 \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateScaleIntToAny_For(typeSrc, typeDst)                                                                 \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, scalefactor_t<typeDst> aDstMin, scalefactor_t<typeDst> aDstMax,                     \
        const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateScaleIntToAnyRound_For(typeSrc, typeDst)                                                            \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, scalefactor_t<typeDst> aDstMin, scalefactor_t<typeDst> aDstMax,                     \
        RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateScaleAnyToInt_For(typeSrc, typeDst)                                                                 \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, scalefactor_t<typeSrc> aSrcMin, scalefactor_t<typeSrc> aSrcMax,                     \
        RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateScaleAnyToAny_For(typeSrc, typeDst)                                                                 \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, scalefactor_t<typeSrc> aSrcMin, scalefactor_t<typeSrc> aSrcMax,                     \
        scalefactor_t<typeDst> aDstMin, scalefactor_t<typeDst> aDstMax, const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateScaleAnyToAnyRound_For(typeSrc, typeDst)                                                            \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, scalefactor_t<typeSrc> aSrcMin, scalefactor_t<typeSrc> aSrcMax,                     \
        scalefactor_t<typeDst> aDstMin, scalefactor_t<typeDst> aDstMax, RoundingMode aRoundingMode,                    \
        const mpp::cuda::StreamCtx &aStreamCtx) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleIntToIntWithAlpha(typeSrc, typeDst)                                                         \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleIntToIntNoAlpha(typeSrc, typeDst)                                                           \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleIntToAnyWithAlpha(typeSrc, typeDst)                                                         \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleIntToAnyNoAlpha(typeSrc, typeDst)                                                           \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleIntToAnyRoundWithAlpha(typeSrc, typeDst)                                                    \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleIntToAnyRoundNoAlpha(typeSrc, typeDst)                                                      \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleAnyToIntWithAlpha(typeSrc, typeDst)                                                         \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleAnyToIntNoAlpha(typeSrc, typeDst)                                                           \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleAnyToAnyWithAlpha(typeSrc, typeDst)                                                         \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleAnyToAnyNoAlpha(typeSrc, typeDst)                                                           \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleAnyToAnyRoundWithAlpha(typeSrc, typeDst)                                                    \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsScaleAnyToAnyRoundNoAlpha(typeSrc, typeDst)                                                      \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

} // namespace mpp::image::cuda