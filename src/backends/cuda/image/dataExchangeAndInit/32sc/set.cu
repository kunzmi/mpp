#if MPP_ENABLE_CUDA_BACKEND

#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetC(32sc);
ForAllChannelsNoAlphaInvokeSetDevC(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
