#if MPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(8u);

ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(8u);

ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(8u);

ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
