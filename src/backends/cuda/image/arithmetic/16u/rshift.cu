#if MPP_ENABLE_CUDA_BACKEND

#include "../rshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRShiftSrcC(16u);
ForAllChannelsWithAlphaInvokeRShiftInplaceC(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
