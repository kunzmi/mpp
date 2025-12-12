#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(64f);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(64f);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(64f);

} // namespace mpp::image::cuda
