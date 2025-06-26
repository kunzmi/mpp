#if MPP_ENABLE_CUDA_BACKEND

#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(64f);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
