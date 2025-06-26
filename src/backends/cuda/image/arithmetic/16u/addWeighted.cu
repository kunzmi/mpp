#if MPP_ENABLE_CUDA_BACKEND

#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(16u);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
