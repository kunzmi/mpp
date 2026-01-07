#include "../../lutToPaletteKernel.h"
#include "../lut_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{
ForAllChannelsWithAlphaInvokeLutPaletteSrc(16s);
ForAllChannelsWithAlphaInvokeLutPaletteInplace(16s);
ForAllChannelsWithAlphaInvokeLutPaletteSrc33(16s);
ForAllChannelsWithAlphaInvokeLutPaletteSrc34A(16s);
ForAllChannelsWithAlphaInvokeLutPaletteSrc4A3(16s);
ForAllChannelsWithAlphaInvokeLutPaletteSrc4A4A(16s);
ForAllChannelsWithAlphaInvokeLutPaletteSrc44(16s);
ForAllChannelsWithAlphaInvokeLutPaletteSrcP(16s);
ForAllChannelsWithAlphaInvokeLutPaletteInplaceP(16s);

ForAllChannelsWithAlphaInvokeLutTrilinearSrc3(16s);
ForAllChannelsWithAlphaInvokeLutTrilinearInplace3(16s);
ForAllChannelsWithAlphaInvokeLutTrilinearSrc4A(16s);
ForAllChannelsWithAlphaInvokeLutTrilinearInplace4A(16s);

template void InvokeLutToPaletteKernelDefault<short>(const int *__restrict__ aX, const int *__restrict__ aY,
                                                     int aLutSize, short *__restrict__ aPalette,
                                                     InterpolationMode aInterpolationMode,
                                                     const mpp::cuda::StreamCtx &aStreamCtx);
} // namespace mpp::image::cuda
