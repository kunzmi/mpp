#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(8s);

ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(8s);

ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(8s);

ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
