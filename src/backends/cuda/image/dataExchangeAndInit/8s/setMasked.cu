#if OPP_ENABLE_CUDA_BACKEND

#include "../setMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(8s);
ForAllChannelsWithAlphaInvokeSetDevCMask(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
