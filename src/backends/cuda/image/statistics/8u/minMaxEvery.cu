#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(8u);

ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(8u);

ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(8u);

ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
