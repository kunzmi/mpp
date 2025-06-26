#if MPP_ENABLE_CUDA_BACKEND

#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(32f);
ForAllChannelsWithAlphaInvokeSetDevC(32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
