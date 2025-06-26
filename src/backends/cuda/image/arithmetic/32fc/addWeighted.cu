#if MPP_ENABLE_CUDA_BACKEND

#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrc(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
