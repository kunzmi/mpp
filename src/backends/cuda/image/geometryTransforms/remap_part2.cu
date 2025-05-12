#if OPP_ENABLE_CUDA_BACKEND

#include "remap_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{
#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeSrc *aDst, size_t aPitchDst, const Pixel32fC2 *aCoordinateMapPtr, \
        size_t aCoordinateMapPitch, InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,    \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

// ForAllChannelsWithAlpha(16f);
// ForAllChannelsWithAlpha(16bf);
// ForAllChannelsWithAlpha(32f);
// ForAllChannelsWithAlpha(64f);
//
// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
// ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeSrc *aDst, size_t aPitchDst,                                      \
        const Pixel32fC1 *aCoordinateMapXPtr, size_t aCoordinateMapXPitch, const Pixel32fC1 *aCoordinateMapYPtr,       \
        size_t aCoordinateMapYPitch, InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,   \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

// ForAllChannelsWithAlpha(16f);
// ForAllChannelsWithAlpha(16bf);
// ForAllChannelsWithAlpha(32f);
// ForAllChannelsWithAlpha(64f);
//
// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
// ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
