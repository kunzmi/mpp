#if MPP_ENABLE_CUDA_BACKEND

#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrcMask(16sc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrcMask(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
