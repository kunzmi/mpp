#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(16f);

ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(16f);

ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(16f);

ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
