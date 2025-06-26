#if MPP_ENABLE_CUDA_BACKEND

#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(8u);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
