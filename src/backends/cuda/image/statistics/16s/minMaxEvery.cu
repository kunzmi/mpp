#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(16s);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(16s);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
