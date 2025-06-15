#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(32u);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(32u);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
