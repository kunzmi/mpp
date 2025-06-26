#if MPP_ENABLE_CUDA_BACKEND

#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrc(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
