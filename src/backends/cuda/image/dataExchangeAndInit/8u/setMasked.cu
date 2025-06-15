#if OPP_ENABLE_CUDA_BACKEND

#include "../setMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(8u);
ForAllChannelsWithAlphaInvokeSetDevCMask(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
