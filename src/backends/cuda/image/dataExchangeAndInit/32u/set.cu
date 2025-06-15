#if OPP_ENABLE_CUDA_BACKEND

#include "../set_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(32u);
ForAllChannelsWithAlphaInvokeSetDevC(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
