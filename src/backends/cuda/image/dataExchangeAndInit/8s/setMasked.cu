#if MPP_ENABLE_CUDA_BACKEND

#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(8s);
ForAllChannelsWithAlphaInvokeSetDevCMask(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
