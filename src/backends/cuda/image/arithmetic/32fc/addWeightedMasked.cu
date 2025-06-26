#if MPP_ENABLE_CUDA_BACKEND

#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrcMask(32fc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrcMask(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
