#if OPP_ENABLE_CUDA_BACKEND

#include "../minMaxEvery_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(32s);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(32s);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
