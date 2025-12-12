#pragma once

namespace mpp::image::cuda
{
// NOLINTBEGIN(bugprone-macro-parentheses,cppcoreguidelines-macro-usage)

#define InstantiateScaleIntToInt_For(typeSrc, typeDst)                                                                 \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx) const;

#define InstantiateScaleIntToAny_For(typeSrc, typeDst)                                                                 \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, complex_basetype_t<pixel_basetype_t<typeDst>> aDstMin,                              \
        complex_basetype_t<pixel_basetype_t<typeDst>> aDstMax, const mpp::cuda::StreamCtx &aStreamCtx) const;

#define InstantiateScaleIntToAnyRound_For(typeSrc, typeDst)                                                            \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, complex_basetype_t<pixel_basetype_t<typeDst>> aDstMin,                              \
        complex_basetype_t<pixel_basetype_t<typeDst>> aDstMax, RoundingMode aRoundingMode,                             \
        const mpp::cuda::StreamCtx &aStreamCtx) const;

#define InstantiateScaleAnyToInt_For(typeSrc, typeDst)                                                                 \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, complex_basetype_t<pixel_basetype_t<typeSrc>> aSrcMin,                              \
        complex_basetype_t<pixel_basetype_t<typeSrc>> aSrcMax, RoundingMode aRoundingMode,                             \
        const mpp::cuda::StreamCtx &aStreamCtx) const;

#define InstantiateScaleAnyToAny_For(typeSrc, typeDst)                                                                 \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, complex_basetype_t<pixel_basetype_t<typeSrc>> aSrcMin,                              \
        complex_basetype_t<pixel_basetype_t<typeSrc>> aSrcMax, complex_basetype_t<pixel_basetype_t<typeDst>> aDstMin,  \
        complex_basetype_t<pixel_basetype_t<typeDst>> aDstMax, const mpp::cuda::StreamCtx &aStreamCtx) const;

#define InstantiateScaleAnyToAnyRound_For(typeSrc, typeDst)                                                            \
    template MPPEXPORT_CUDAI ImageView<typeDst> &ImageView<typeSrc>::Scale<typeDst>(                                   \
        ImageView<typeDst> & aDst, scalefactor_t<typeSrc> aSrcMin, scalefactor_t<typeSrc> aSrcMax,                     \
        scalefactor_t<typeDst> aDstMin, scalefactor_t<typeDst> aDstMax, RoundingMode aRoundingMode,                    \
        const mpp::cuda::StreamCtx &aStreamCtx) const;

#define ForAllChannelsScaleIntToIntWithAlpha(typeSrc, typeDst)                                                         \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#define ForAllChannelsScaleIntToIntNoAlpha(typeSrc, typeDst)                                                           \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleIntToInt_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsScaleIntToAnyWithAlpha(typeSrc, typeDst)                                                         \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#define ForAllChannelsScaleIntToAnyNoAlpha(typeSrc, typeDst)                                                           \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleIntToAny_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsScaleIntToAnyRoundWithAlpha(typeSrc, typeDst)                                                    \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#define ForAllChannelsScaleIntToAnyRoundNoAlpha(typeSrc, typeDst)                                                      \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateScaleIntToAnyRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsScaleAnyToIntWithAlpha(typeSrc, typeDst)                                                         \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#define ForAllChannelsScaleAnyToIntNoAlpha(typeSrc, typeDst)                                                           \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleAnyToInt_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsScaleAnyToAnyWithAlpha(typeSrc, typeDst)                                                         \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#define ForAllChannelsScaleAnyToAnyNoAlpha(typeSrc, typeDst)                                                           \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateScaleAnyToAny_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsScaleAnyToAnyRoundWithAlpha(typeSrc, typeDst)                                                    \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#define ForAllChannelsScaleAnyToAnyRoundNoAlpha(typeSrc, typeDst)                                                      \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateScaleAnyToAnyRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

// NOLINTEND(bugprone-macro-parentheses,cppcoreguidelines-macro-usage)
} // namespace mpp::image::cuda