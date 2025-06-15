#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(64f);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(64f);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
