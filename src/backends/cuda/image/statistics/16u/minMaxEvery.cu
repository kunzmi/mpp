#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(16u);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(16u);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
