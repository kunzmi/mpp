#include "../../lutAcceleratorKernel.h"
#include "../lut_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{
ForAllChannelsWithAlphaInvokeLutSrc1(16f);
ForAllChannelsWithAlphaInvokeLutInplace1(16f);
ForAllChannelsWithAlphaInvokeLutSrc(16f);
ForAllChannelsWithAlphaInvokeLutInplace(16f);
} // namespace mpp::image::cuda
