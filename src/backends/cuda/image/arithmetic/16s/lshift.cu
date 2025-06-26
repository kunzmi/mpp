#if MPP_ENABLE_CUDA_BACKEND

#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(16s);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
