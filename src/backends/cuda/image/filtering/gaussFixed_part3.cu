#if OPP_ENABLE_CUDA_BACKEND

#include "gaussFixed_impl.h"

namespace opp::image::cuda
{

#pragma region Instantiate

#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeGaussFixed<typeSrcIsTypeDst, typeSrcIsTypeDst>(                                                \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        MaskSize aMaskSize, BorderType aBorderType, const typeSrcIsTypeDst &aConstant,                                 \
        const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,                \
        const StreamCtx &aStreamCtx);

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
//
// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);
//
// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
// ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
