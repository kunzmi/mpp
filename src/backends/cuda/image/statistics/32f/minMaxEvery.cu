#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(32f);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(32f);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(32f);

} // namespace mpp::image::cuda
