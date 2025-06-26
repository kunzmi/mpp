#if MPP_ENABLE_CUDA_BACKEND

#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(16u);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
