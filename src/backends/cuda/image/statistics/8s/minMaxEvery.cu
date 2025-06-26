#if MPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(8s);

ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(8s);

ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(8s);

ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
