#include "../../lutToPaletteKernel.h"
#include "../lut_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{
ForAllChannelsWithAlphaInvokeLutPaletteSrc(8u);
ForAllChannelsWithAlphaInvokeLutPaletteInplace(8u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc33(8u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc34A(8u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc4A3(8u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc4A4A(8u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc44(8u);
ForAllChannelsWithAlphaInvokeLutPaletteSrcP(8u);
ForAllChannelsWithAlphaInvokeLutPaletteInplaceP(8u);

ForAllChannelsWithAlphaInvokeLutTrilinearSrc3(8u);
ForAllChannelsWithAlphaInvokeLutTrilinearInplace3(8u);
ForAllChannelsWithAlphaInvokeLutTrilinearSrc4A(8u);
ForAllChannelsWithAlphaInvokeLutTrilinearInplace4A(8u);

template void InvokeLutToPaletteKernelDefault<byte>(const int *__restrict__ aX, const int *__restrict__ aY,
                                                    int aLutSize, byte *__restrict__ aPalette,
                                                    InterpolationMode aInterpolationMode,
                                                    const mpp::cuda::StreamCtx &aStreamCtx);
} // namespace mpp::image::cuda
