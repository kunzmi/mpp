#include "../../lutAcceleratorKernel.h"
#include "../lut_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLutSrc1(16bf);
ForAllChannelsWithAlphaInvokeLutInplace1(16bf);
ForAllChannelsWithAlphaInvokeLutSrc(16bf);
ForAllChannelsWithAlphaInvokeLutInplace(16bf);
} // namespace mpp::image::cuda
