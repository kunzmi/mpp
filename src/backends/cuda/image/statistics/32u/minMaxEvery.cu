#if MPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(32u);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(32u);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
