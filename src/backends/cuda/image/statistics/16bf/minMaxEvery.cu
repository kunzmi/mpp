#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(16bf);

ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(16bf);

ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(16bf);

ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
