#if MPP_ENABLE_CUDA_BACKEND

#include "../rshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRShiftSrcC(8u);
ForAllChannelsWithAlphaInvokeRShiftInplaceC(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
