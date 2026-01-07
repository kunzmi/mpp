#include "../../lutToPaletteKernel.h"
#include "../lut_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{
ForAllChannelsWithAlphaInvokeLutPaletteSrc(16u);
ForAllChannelsWithAlphaInvokeLutPaletteInplace(16u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc33(16u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc34A(16u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc4A3(16u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc4A4A(16u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc44(16u);
ForAllChannelsWithAlphaInvokeLutPaletteSrcP(16u);
ForAllChannelsWithAlphaInvokeLutPaletteInplaceP(16u);

ForAllChannelsWithAlphaInvokeLutPaletteSrc8uC1(16u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc8uC3(16u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc8uC4(16u);
ForAllChannelsWithAlphaInvokeLutPaletteSrc8uC4A(16u);

ForAllChannelsWithAlphaInvokeLutTrilinearSrc3(16u);
ForAllChannelsWithAlphaInvokeLutTrilinearInplace3(16u);
ForAllChannelsWithAlphaInvokeLutTrilinearSrc4A(16u);
ForAllChannelsWithAlphaInvokeLutTrilinearInplace4A(16u);

template void InvokeLutToPaletteKernelDefault<ushort>(const int *__restrict__ aX, const int *__restrict__ aY,
                                                      int aLutSize, ushort *__restrict__ aPalette,
                                                      InterpolationMode aInterpolationMode,
                                                      const mpp::cuda::StreamCtx &aStreamCtx);
} // namespace mpp::image::cuda
