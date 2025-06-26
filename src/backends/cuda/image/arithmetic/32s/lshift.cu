#if MPP_ENABLE_CUDA_BACKEND

#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(32s);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
