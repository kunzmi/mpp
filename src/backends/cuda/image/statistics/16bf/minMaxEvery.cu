#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(16bf);

ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(16bf);

ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(16bf);

ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(16bf);

} // namespace mpp::image::cuda
