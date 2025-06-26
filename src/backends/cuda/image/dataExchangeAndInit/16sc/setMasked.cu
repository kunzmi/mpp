#if MPP_ENABLE_CUDA_BACKEND

#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetCMask(16sc);
ForAllChannelsNoAlphaInvokeSetDevCMask(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
