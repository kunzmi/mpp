#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(32f);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(32f);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
