#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(16u);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(16u);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(16u);

} // namespace mpp::image::cuda
