#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(16f);

ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(16f);

ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(16f);

ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(16f);

} // namespace mpp::image::cuda
