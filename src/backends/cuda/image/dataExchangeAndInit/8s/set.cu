#if MPP_ENABLE_CUDA_BACKEND

#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(8s);
ForAllChannelsWithAlphaInvokeSetDevC(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
