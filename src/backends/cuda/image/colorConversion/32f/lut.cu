#include "../../lutAcceleratorKernel.h"
#include "../lut_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLutSrc1(32f);
ForAllChannelsWithAlphaInvokeLutInplace1(32f);
ForAllChannelsWithAlphaInvokeLutSrc(32f);
ForAllChannelsWithAlphaInvokeLutInplace(32f);

ForAllChannelsWithAlphaInvokeLutTrilinearSrc3(32f);
ForAllChannelsWithAlphaInvokeLutTrilinearInplace3(32f);
ForAllChannelsWithAlphaInvokeLutTrilinearSrc4A(32f);
ForAllChannelsWithAlphaInvokeLutTrilinearInplace4A(32f);

} // namespace mpp::image::cuda
